import torch
import torch.nn.functional as F
import torch.nn as nn
import torchaudio
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from utils import init_weights, get_padding

LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)

class PreNet(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, num_mels=80, sampling_rate = 22050, f_min=0, f_max=8000,  win_length=None):

        super(PreNet, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length if win_length else filter_length
        self.forward_transform = None
        self.pad_amount = int(self.filter_length / 2)
        self.f_min=f_min
        self.f_max=f_max
        self.num_mels=num_mels
        self.sampling_rate=sampling_rate
        scale = self.filter_length / self.hop_length
        # fourier_basis = np.fft.fft(np.eye(self.filter_length))
        fourier_basis = torch.fft.fft(torch.eye(self.filter_length))
        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = torch.cat([torch.real(fourier_basis[:cutoff, :]),
                                   torch.imag(fourier_basis[:cutoff, :])],dim=0)
        self.dft = nn.Conv1d(in_channels =1 , out_channels =self.filter_length+2 , kernel_size = self.filter_length, stride = self.hop_length,dtype=torch.complex64, bias = False) 
        self.mel = nn.Conv1d(in_channels=self.filter_length//2+1, out_channels=self.num_mels, kernel_size=1, stride=1, bias=False)
        self.prelu = nn.PReLU(init=0)
        mel_basis = torchaudio.transforms.MelScale(n_mels=self.num_mels, sample_rate=self.sampling_rate, f_min=self.f_min, f_max=self.f_max, n_stft=self.filter_length//2+1, norm='slaney', mel_scale='slaney')(torch.eye(self.filter_length//2+1)).unsqueeze(2)
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        assert(filter_length >= self.win_length)

        fft_window = torch.hann_window(self.win_length)
        # fft_window = pad_center(fft_window, filter_length)

        forward_basis *= fft_window

        self.dft.weight.data = forward_basis
        self.mel.weight.data = mel_basis

        # self.register_buffer('forward_basis', forward_basis.float())

    def magnitude(self, input_data):
        num_batches = input_data.shape[0]
        num_samples = input_data.shape[-1]

        if input_data.dim() == 2:
            input_data = input_data.unsqueeze(1)
            
        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        # input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data,
            (int((self.win_length-self.hop_length)/2), int((self.win_length-self.hop_length)/2)),
            mode='reflect')

        forward_transform = self.dft(input_data)
        # cutoff = int((self.filter_length / 2) + 1)
        # real_part = forward_transform[:, :cutoff, :]
        # imag_part = forward_transform[:, cutoff:, :]
        # magnitude = torch.sqrt(real_part**2 + imag_part**2 +(1e-9))
        magnitude = torch.linalg.norm(forward_transform.reshape(forward_transform.shape[0],2,-1,forward_transform.shape[-1]), dim=1)
        return magnitude

    def log_magnitude(self, input_data):
        return torch.log(torch.clamp(self.magnitude(input_data), min=1e-5))

    def log_mel(self, input_data):
        magnitude = self.magnitude(input_data)
        return torch.log(torch.clamp(self.prelu(self.mel(magnitude)), min=1e-5))

    def forward(self, input_data):
        magnitude = self.magnitude(input_data)
        return torch.log(torch.clamp(self.prelu(self.mel(magnitude)), min=1e-5))

    def magnitude_nograd(self, input_data):
        num_batches = input_data.shape[0]
        num_samples = input_data.shape[-1]

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        # input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data,
            (int((self.win_length-self.hop_length)/2), int((self.win_length-self.hop_length)/2)),
            mode='reflect')
        # input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            self.dft.weight.data.clone().detach(),
            stride=self.hop_length,
            padding=0)

        magnitude = torch.linalg.norm(forward_transform.reshape(forward_transform.shape[0],2,-1,forward_transform.shape[-1]), dim=1)
        return magnitude

    def log_magnitude_nograd(self, input_data):
        return torch.log(torch.clamp(self.magnitude_nograd(input_data), min=1e-5))

    def log_mel_nograd(self, input_data):
        magnitude = self.magnitude_nograd(input_data)
        mel_spec = F.conv1d(
            magnitude,
            self.mel.weight.data.clone().detach(),
            stride=1,
            padding=0)
        ret = torch.log(torch.clamp(mel_spec, min=1e-5))
        return ret


class Generator(torch.nn.Module):
    def __init__(self, h, prenet_copy=None):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        if prenet_copy:
            self.pre_net = prenet_copy
        else:
            self.pre_net = PreNet(filter_length=h.win_size, hop_length=h.hop_size, num_mels=h.num_mels, sampling_rate = h.sampling_rate, f_min=h.fmin, f_max=h.fmax,  win_length=None)
        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.pre_net(x)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward_one(self, y):
        y_d_rs = []
        fmap_rs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)

        return y_d_rs, fmap_rs

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward_one(self, y):
        y_d_rs = []
        fmap_rs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
            y_d_r, fmap_r = d(y)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)

        return y_d_rs, fmap_rs

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DownSample_Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DownSample_Discriminator, self).__init__()

        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.InstanceNorm2d(num_features=out_channels,
                                                         affine=True))
        self.convLayerGates = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                      out_channels=out_channels,
                                                      kernel_size=kernel_size,
                                                      stride=stride,
                                                      padding=padding),
                                            nn.InstanceNorm2d(num_features=out_channels,
                                                              affine=True))

    def forward(self, input):
        # GLU
        return self.convLayer(input) * torch.sigmoid(self.convLayerGates(input))


class PatchGANDiscriminator(nn.Module):
    def __init__(self):
        super(PatchGANDiscriminator, self).__init__()

        self.convLayer1 = nn.Conv2d(in_channels=1,
                                    out_channels=128,
                                    kernel_size=[3, 3],
                                    stride=[1, 1])
        self.convLayer1_gates = nn.Conv2d(in_channels=1,
                                          out_channels=128,
                                          kernel_size=[3, 3],
                                          stride=[1, 1])

        # Note: Kernel Size have been modified in the PyTorch implementation
        # compared to the actual paper, as to retain dimensionality. Unlike,
        # TensorFlow, PyTorch doesn't have padding='same', hence, kernel sizes
        # were altered to retain the dimensionality after each layer

        # DownSample Layer
        self.downSample1 = DownSample_Discriminator(in_channels=128,
                                                    out_channels=256,
                                                    kernel_size=[3, 3],
                                                    stride=[2, 2],
                                                    padding=0)

        self.downSample2 = DownSample_Discriminator(in_channels=256,
                                                    out_channels=512,
                                                    kernel_size=[3, 3],
                                                    stride=[2, 2],
                                                    padding=0)

        self.downSample3 = DownSample_Discriminator(in_channels=512,
                                                    out_channels=1024,
                                                    kernel_size=[3, 3],
                                                    stride=[2, 2],
                                                    padding=0)

        self.downSample4 = DownSample_Discriminator(in_channels=1024,
                                                    out_channels=1024,
                                                    kernel_size=[1, 5],
                                                    stride=[1, 1],
                                                    padding=[0, 2])

        # Fully Connected Layer
        self.fc = nn.Linear(in_features=1024,
                            out_features=1)

        # output Layer
        self.output_layer = nn.Conv2d(in_channels=1024,
                                      out_channels=1,
                                      kernel_size=[1, 3],
                                      stride=[1, 1],
                                      padding=[0, 1])

    # def downSample(self, in_channels, out_channels, kernel_size, stride, padding):
    #     convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
    #                                         out_channels=out_channels,
    #                                         kernel_size=kernel_size,
    #                                         stride=stride,
    #                                         padding=padding),
    #                               nn.InstanceNorm2d(num_features=out_channels,
    #                                                 affine=True),
    #                               GLU())
    #     return convLayer

    def forward(self, input):
        # input has shape [batch_size, num_features, time]
        # discriminator requires shape [batchSize, 1, num_features, time]
        input = input.unsqueeze(1)
        # GLU
        pad_input = nn.ZeroPad2d((1, 1, 1, 1))
        layer1 = self.convLayer1(
            pad_input(input)) * torch.sigmoid(self.convLayer1_gates(pad_input(input)))

        pad_input = nn.ZeroPad2d((1, 0, 1, 0))
        downSample1 = self.downSample1(pad_input(layer1))

        pad_input = nn.ZeroPad2d((1, 0, 1, 0))
        downSample2 = self.downSample2(pad_input(downSample1))

        pad_input = nn.ZeroPad2d((1, 0, 1, 0))
        downSample3 = self.downSample3(pad_input(downSample2))

        downSample4 = self.downSample4(downSample3)
        downSample4 = self.output_layer(downSample4)

        downSample4 = downSample4.contiguous().permute(0, 2, 3, 1).contiguous()
        # fc = torch.sigmoid(self.fc(downSample3))
        # Taking off sigmoid layer to avoid vanishing gradient problem
        #fc = self.fc(downSample4)
        fc = torch.sigmoid(downSample4)
        return fc



def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

