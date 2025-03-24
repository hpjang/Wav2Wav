import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, PatchGANDiscriminator, feature_loss, generator_loss,\
    discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
import numpy as np
import random
from queue import PriorityQueue

def calc_current_lr(step, start_lr=0.0001, algorithm="linear", ratio=0.001):
    if (algorithm=="linear"):
        last_step = 100000
        d = (start_lr - ratio*start_lr)/last_step
        lr = max(0, start_lr-step*d)
    elif (algorithm=="exponential"):
        lr = start_lr*pow(ratio, step)
    return lr

def set_seed(step, seed):
    torch.cuda.manual_seed(step+seed)
    torch.manual_seed(step+seed)
    np.random.seed(step+seed)
    random.seed(step+seed)

def train(rank, a, h):
    loss_q = PriorityQueue()
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generatorM = Generator(h).to(device)
    generatorF = Generator(h, generatorM.pre_net).to(device)
    mpdM = MultiPeriodDiscriminator().to(device)
    msdM = MultiScaleDiscriminator().to(device)
    mpdF = MultiPeriodDiscriminator().to(device)
    msdF = MultiScaleDiscriminator().to(device)
    pgdM = PatchGANDiscriminator().to(device)
    pgdF = PatchGANDiscriminator().to(device)
    last_epoch = 0

    if rank == 0:
        # print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = steps = h.starting_step
    # if cp_g is None or cp_do is None:
    #     state_dict_do = None
    #     last_epoch = -1
    # else:
    #     state_dict_g = load_checkpoint(cp_g, device)
    #     state_dict_do = load_checkpoint(cp_do, device)

    if (a.load_vocoder=="True"):
        state_dict_gM = load_checkpoint(a.vocoder_load_gen_path, device)
        generatorM.load_state_dict(state_dict_gM['generator'], strict=False)
        state_dict_gF = load_checkpoint(a.vocoder_load_gen_path, device)
        generatorF.load_state_dict(state_dict_gF['generator'], strict=False)
        state_dict_doM = load_checkpoint(a.vocoder_load_disc_path, device)
        mpdM.load_state_dict(state_dict_doM['mpd'])
        msdM.load_state_dict(state_dict_doM['msd'])
        state_dict_doF = load_checkpoint(a.vocoder_load_disc_path, device)
        mpdF.load_state_dict(state_dict_doF['mpd'])
        msdF.load_state_dict(state_dict_doF['msd'])

    # if h.num_gpus > 1:
    #     generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
    #     mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
    #     msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    optim_gM = torch.optim.AdamW(generatorM.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_dM = torch.optim.AdamW(itertools.chain(msdM.parameters(), mpdM.parameters(), pgdM.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_gF = torch.optim.AdamW(generatorF.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_dF = torch.optim.AdamW(itertools.chain(msdF.parameters(), mpdF.parameters(), pgdF.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    # if state_dict_do is not None:
    #     optim_g.load_state_dict(state_dict_do['optim_g'])
    #     optim_d.load_state_dict(state_dict_do['optim_d'])


    training_filelistM, training_filelistF, validation_filelistM,  validation_filelistF = get_dataset_filelist(a)

    trainsetM = MelDataset(training_filelistM, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)

    trainsetF = MelDataset(training_filelistF, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)

    # train_sampler = DistributedSampler(trainsetM) if h.num_gpus > 1 else None

    train_loaderM = DataLoader(trainsetM, num_workers=h.num_workers, shuffle=False,
                              sampler=None,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)
    train_loaderF = DataLoader(trainsetF, num_workers=h.num_workers, shuffle=False,
                              sampler=None,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    if rank == 0:
        validsetM = MelDataset(validation_filelistM, h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
                              base_mels_path=a.input_mels_dir)
        validation_loaderM = DataLoader(validsetM, num_workers=0, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=False)

        validsetF = MelDataset(validation_filelistF, h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
                              base_mels_path=a.input_mels_dir)
        validation_loaderF = DataLoader(validsetF, num_workers=0, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=False)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generatorM.train()
    generatorF.train()
    mpdM.train()
    mpdF.train()
    msdM.train()
    msdF.train()
    pgdM.train()
    pgdF.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for (i, batchM), (j, batchF) in zip(enumerate(train_loaderM), enumerate(train_loaderF)):
            optim_gM.param_groups[0]['lr'] = calc_current_lr(steps, start_lr=h.learning_rate, algorithm="linear", ratio=h.lr_decay)
            optim_gF.param_groups[0]['lr'] = calc_current_lr(steps, start_lr=h.learning_rate, algorithm="linear", ratio=h.lr_decay)
            optim_dM.param_groups[0]['lr'] = calc_current_lr(steps, start_lr=h.learning_rate, algorithm="linear", ratio=h.lr_decay)
            optim_dF.param_groups[0]['lr'] = calc_current_lr(steps, start_lr=h.learning_rate, algorithm="linear", ratio=h.lr_decay)
            set_seed(steps, h.seed)

            if rank == 0:
                start_b = time.time()
            # x: mel spectorgram, y: raw audio
            xM, yM, _, yM_mel = batchM
            xM = torch.autograd.Variable(xM.to(device, non_blocking=True))
            yM = torch.autograd.Variable(yM.to(device, non_blocking=True))
            yM_mel = torch.autograd.Variable(yM_mel.to(device, non_blocking=True))
            yM = yM.unsqueeze(1)

            xF, yF, _, yF_mel = batchF
            xF = torch.autograd.Variable(xF.to(device, non_blocking=True))
            yF = torch.autograd.Variable(yF.to(device, non_blocking=True))
            yF_mel = torch.autograd.Variable(yF_mel.to(device, non_blocking=True))
            yF = yF.unsqueeze(1)

            if (steps < a.identity_loss_due):
                id_lambda=a.identity_loss_lambda
            else:
                id_lambda=0

            optim_gF.zero_grad()
            optim_gM.zero_grad()

            yMyF = generatorF(yM) # fake audio
            yMyFyM = generatorM(yMyF) # recon audio
            yFyM = generatorM(yF) # fake audio
            yFyMyF = generatorF(yFyM) # recon audio

            yFyF = generatorF(yF) # identity
            yMyM = generatorM(yM) # identity

            yMxF = generatorF.pre_net.log_mel_nograd(yMyF)
            yFxM = generatorM.pre_net.log_mel_nograd(yFyM)
            yFyMxF = generatorF.pre_net.log_mel_nograd(yFyMyF)
            yMyFxM = generatorM.pre_net.log_mel_nograd(yMyFyM)
            d_fake_F = pgdF(yMxF)
            d_fake_M = pgdM(yFxM)
            d_fake_cycle_F = pgdF(yMyFxM)
            d_fake_cycle_M = pgdM(yFyMxF)

            generator_loss(d_fake_M)

            g_loss_F2M = generator_loss(d_fake_M)[0]
            g_loss_M2F = generator_loss(d_fake_F)[0]

            generator_loss(d_fake_cycle_F)
            g_loss_F2M2F = generator_loss(d_fake_cycle_F)[0]
            g_loss_M2F2M = generator_loss(d_fake_cycle_M)[0]


            yF_p = generatorF.pre_net.log_magnitude_nograd(yF).clone().detach()
            yM_p = generatorM.pre_net.log_magnitude_nograd(yM).clone().detach()
            yFyF_p = generatorF.pre_net.log_magnitude_nograd(yFyF)
            yMyM_p = generatorM.pre_net.log_magnitude_nograd(yMyM)
            yFyMyF_p = generatorF.pre_net.log_magnitude_nograd(yFyMyF)
            yMyFyM_p = generatorM.pre_net.log_magnitude_nograd(yMyFyM)

            cycle_loss_F = F.l1_loss(yF_p, yFyMyF_p)
            cycle_loss_M = F.l1_loss(yM_p, yMyFyM_p)
            cycle_loss = cycle_loss_F + cycle_loss_M


            id_loss_F = F.l1_loss(yF_p, yFyF_p)
            id_loss_M = F.l1_loss(yM_p, yMyM_p)
            id_loss = id_loss_F + id_loss_M

            _, Fy_df_hat_g, Ffmap_f_r, Ffmap_f_g = mpdF(yF, yFyMyF)
            _, Fy_ds_hat_g, Ffmap_s_r, Ffmap_s_g = msdF(yF, yFyMyF)
            Floss_fm_f = feature_loss(Ffmap_f_r, Ffmap_f_g)
            Floss_fm_s = feature_loss(Ffmap_s_r, Ffmap_s_g)
            Floss_gen_f, _ = generator_loss(Fy_df_hat_g)
            Floss_gen_s, _ = generator_loss(Fy_ds_hat_g)
            _, My_df_hat_g, Mfmap_f_r, Mfmap_f_g = mpdM(yM, yMyFyM)
            _, My_ds_hat_g, Mfmap_s_r, Mfmap_s_g = msdM(yM, yMyFyM)
            Mloss_fm_f = feature_loss(Mfmap_f_r, Mfmap_f_g)
            Mloss_fm_s = feature_loss(Mfmap_s_r, Mfmap_s_g)
            Mloss_gen_f, _ = generator_loss(My_df_hat_g)
            Mloss_gen_s, _ = generator_loss(My_ds_hat_g)

            g_loss = g_loss_F2M + g_loss_M2F + g_loss_F2M2F + g_loss_M2F2M + \
                    Floss_gen_f + Floss_gen_s + Mloss_gen_f + Mloss_gen_s + \
                    a.cycle_loss_lambda*cycle_loss + a.identity_loss_lambda*id_loss + \
                    a.fm_loss_lambda*(Floss_fm_f + Floss_fm_s + Mloss_fm_f + Mloss_fm_s)

            g_loss.backward()
            optim_gF.step()
            optim_gM.step()




            optim_dF.zero_grad()
            optim_dM.zero_grad()

            d_real_F = pgdF(xF) # d_real_M: D(x)
            d_real_M = pgdM(xM) # d_real_M: D(x)

            yMyF = generatorF(yM) # fake audio
            yMyFyM = generatorM(yMyF) # recon audio
            yFyM = generatorM(yF) # fake audio
            yFyMyF = generatorF(yFyM) # recon audio

            yMxF = generatorF.pre_net.log_mel_nograd(yMyF)
            yFxM = generatorM.pre_net.log_mel_nograd(yFyM)
            yFyMxF = generatorF.pre_net.log_mel_nograd(yFyMyF)
            yMyFxM = generatorM.pre_net.log_mel_nograd(yMyFyM)

            d_fake_F = pgdF(yMxF)
            d_fake_M = pgdM(yFxM)

            d_fake_cycle_F = pgdF(yMyFxM)
            d_fake_cycle_M = pgdM(yFyMxF)

            d_loss_F = (discriminator_loss(d_real_F, d_fake_F)[0] + discriminator_loss(d_real_F, d_fake_cycle_F)[0])/2
            d_loss_M = (discriminator_loss(d_real_M, d_fake_M)[0] + discriminator_loss(d_real_M, d_fake_cycle_M)[0])/2


            # MPDF
            Fy_df_hat_r, Fy_df_hat_g, _, _ = mpdF(yF, yFyMyF.detach())
            Floss_disc_f, _, _ = discriminator_loss(Fy_df_hat_r, Fy_df_hat_g)

            # MSDF
            Fy_ds_hat_r, Fy_ds_hat_g, _, _ = msdF(yF, yFyMyF.detach())
            Floss_disc_s, _, _ = discriminator_loss(Fy_ds_hat_r, Fy_ds_hat_g)

            # MPDM
            My_df_hat_r, My_df_hat_g, _, _ = mpdM(yM, yMyFyM.detach())
            Mloss_disc_f, _, _ = discriminator_loss(My_df_hat_r, My_df_hat_g)

            # MSDM
            My_ds_hat_r, My_ds_hat_g, _, _ = msdM(yM, yMyFyM.detach())
            Mloss_disc_s, _, _ = discriminator_loss(My_ds_hat_r, My_ds_hat_g)

            loss_disc_all = Floss_disc_s + Floss_disc_f + d_loss_F + Mloss_disc_s + Mloss_disc_f + d_loss_M

            loss_disc_all.backward()
            optim_dF.step()
            optim_dM.step()



            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = ((F.l1_loss(yM_mel, yMyFxM) + F.l1_loss(yF_mel, yFyMxF))/2).item()

                    print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, g_loss, mel_error, time.time() - start_b))

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", g_loss, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)
                    sw.add_scalar("training/learning_rate", optim_gM.param_groups[0]['lr'], steps)

                # Validation
                if steps % a.validation_interval == 0:  # and steps != 0:
                    generatorM.eval()
                    generatorF.eval()
                    torch.cuda.empty_cache()
                    val_err_totF = 0
                    val_err_totM = 0
                    with torch.no_grad():
                        for (i, batchM), (j, batchF) in zip(enumerate(validation_loaderM), enumerate(validation_loaderF)):
                            xM, yM, _, yM_mel = batchM
                            xM = torch.autograd.Variable(xM.to(device, non_blocking=True))
                            yM = torch.autograd.Variable(yM.to(device, non_blocking=True))
                            yM_mel = torch.autograd.Variable(yM_mel.to(device, non_blocking=True))
                            yM = yM.unsqueeze(1)

                            xF, yF, _, yF_mel = batchF
                            xF = torch.autograd.Variable(xF.to(device, non_blocking=True))
                            yF = torch.autograd.Variable(yF.to(device, non_blocking=True))
                            yF_mel = torch.autograd.Variable(yF_mel.to(device, non_blocking=True))
                            yF = yF.unsqueeze(1)

                            yMyF = generatorF(yM) # fake audio
                            yMyFyM = generatorM(yMyF.detach()) # recon audio
                            yMyFxM = generatorM.pre_net(yMyFyM)
                            # yMyFxM = mel_spectrogram(yMyFyM.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                            #                             h.fmin, h.fmax_for_loss)
                            val_err_totM += F.l1_loss(yM_mel, yMyFxM).item()


                            yFyM = generatorM(yF) # fake audio
                            yFyMyF = generatorF(yFyM.detach()) # recon audio
                            yFyMxF = generatorF.pre_net(yFyMyF)
                            # yFyMxF = mel_spectrogram(yFyMyF.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                            #                                 h.fmin, h.fmax_for_loss)
                            val_err_totF += F.l1_loss(yF_mel, yFyMxF).item()

                        total_valid_loss = (val_err_totF + val_err_totM)/2
                        sw.add_scalar("validation/cycle-consistency_loss_A2B2A", val_err_totM / (j+1), steps)
                        sw.add_scalar("validation/cycle-consistency_loss_B2A2B", val_err_totF / (j+1), steps)
                        sw.add_scalar("validation/cycle-consistency_loss_total", total_valid_loss / (j+1), steps)

                        # if loss_q.qsize() <= 5:
                        #     # checkpointing
                        #     checkpoint_path = "{}/gM_{:08d}".format(a.checkpoint_path, steps)
                        #     save_checkpoint(checkpoint_path,
                        #                     {'generator': (generatorM.module if h.num_gpus > 1 else generatorM).state_dict()})
                        #     checkpoint_path = "{}/doM_{:08d}".format(a.checkpoint_path, steps)
                        #     save_checkpoint(checkpoint_path, 
                        #                     {'mpd': (mpdM.module if h.num_gpus > 1
                        #                                         else mpdM).state_dict(),
                        #                     'msd': (msdM.module if h.num_gpus > 1
                        #                                         else msdM).state_dict(),
                        #                     'pgd' : pgdM.state_dict(),
                        #                     'optim_g': optim_gM.state_dict(), 'optim_d': optim_dM.state_dict(), 'steps': steps,
                        #                     'epoch': epoch})

                        #     checkpoint_path = "{}/gF_{:08d}".format(a.checkpoint_path, steps)
                        #     save_checkpoint(checkpoint_path,
                        #                     {'generator': (generatorF.module if h.num_gpus > 1 else generatorF).state_dict()})
                        #     checkpoint_path = "{}/doF_{:08d}".format(a.checkpoint_path, steps)
                        #     save_checkpoint(checkpoint_path, 
                        #                     {'mpd': (mpdF.module if h.num_gpus > 1
                        #                                         else mpdF).state_dict(),
                        #                     'msd': (msdF.module if h.num_gpus > 1
                        #                                         else msdF).state_dict(),
                        #                     'pgd' : pgdF.state_dict(),
                        #                     'optim_g': optim_gF.state_dict(), 'optim_d': optim_dF.state_dict(), 'steps': steps,
                        #                     'epoch': epoch})
                        #     loss_q.put((-1*total_valid_loss, steps))

                        # elif -1*loss_q.queue[0][0] > total_valid_loss:
                        #     # checkpointing
                        #     checkpoint_path = "{}/gM_{:08d}".format(a.checkpoint_path, steps)
                        #     save_checkpoint(checkpoint_path,
                        #                     {'generator': (generatorM.module if h.num_gpus > 1 else generatorM).state_dict()})
                        #     checkpoint_path = "{}/doM_{:08d}".format(a.checkpoint_path, steps)
                        #     save_checkpoint(checkpoint_path, 
                        #                     {'mpd': (mpdM.module if h.num_gpus > 1
                        #                                         else mpdM).state_dict(),
                        #                     'msd': (msdM.module if h.num_gpus > 1
                        #                                         else msdM).state_dict(),
                        #                     'pgd' : pgdM.state_dict(),
                        #                     'optim_g': optim_gM.state_dict(), 'optim_d': optim_dM.state_dict(), 'steps': steps,
                        #                     'epoch': epoch})

                        #     checkpoint_path = "{}/gF_{:08d}".format(a.checkpoint_path, steps)
                        #     save_checkpoint(checkpoint_path,
                        #                     {'generator': (generatorF.module if h.num_gpus > 1 else generatorF).state_dict()})
                        #     checkpoint_path = "{}/doF_{:08d}".format(a.checkpoint_path, steps)
                        #     save_checkpoint(checkpoint_path, 
                        #                     {'mpd': (mpdF.module if h.num_gpus > 1
                        #                                         else mpdF).state_dict(),
                        #                     'msd': (msdF.module if h.num_gpus > 1
                        #                                         else msdF).state_dict(),
                        #                     'pgd' : pgdF.state_dict(),
                        #                     'optim_g': optim_gF.state_dict(), 'optim_d': optim_dF.state_dict(), 'steps': steps,
                        #                     'epoch': epoch})
                            
                        #     os.remove("{}/gM_{:08d}".format(a.checkpoint_path, loss_q.queue[0][1]))
                        #     os.remove("{}/doM_{:08d}".format(a.checkpoint_path, loss_q.queue[0][1]))
                        #     os.remove("{}/gF_{:08d}".format(a.checkpoint_path, loss_q.queue[0][1]))
                        #     os.remove("{}/doF_{:08d}".format(a.checkpoint_path, loss_q.queue[0][1]))
                        #     loss_q.get()
                        #     loss_q.put((-1*total_valid_loss, steps))

                if steps%a.checkpoint_interval == 0 and steps != 0:
                # if steps == 95400:
                    checkpoint_path = "{}/gM_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generatorM.module if h.num_gpus > 1 else generatorM).state_dict()})
                    # checkpoint_path = "{}/doM_{:08d}".format(a.checkpoint_path, steps)
                    # save_checkpoint(checkpoint_path, 
                    #                 {'mpd': (mpdM.module if h.num_gpus > 1
                    #                                         else mpdM).state_dict(),
                    #                     'msd': (msdM.module if h.num_gpus > 1
                    #                                         else msdM).state_dict(),
                    #                     'pgd' : pgdM.state_dict(),
                    #                     'optim_g': optim_gM.state_dict(), 'optim_d': optim_dM.state_dict(), 'steps': steps,
                    #                     'epoch': epoch})

                    checkpoint_path = "{}/gF_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generatorF.module if h.num_gpus > 1 else generatorF).state_dict()})
                    # checkpoint_path = "{}/doF_{:08d}".format(a.checkpoint_path, steps)
                    # save_checkpoint(checkpoint_path, 
                    #                 {'mpd': (mpdF.module if h.num_gpus > 1
                    #                                         else mpdF).state_dict(),
                    #                     'msd': (msdF.module if h.num_gpus > 1
                    #                                         else msdF).state_dict(),
                    #                     'pgd' : pgdF.state_dict(),
                    #                     'optim_g': optim_gF.state_dict(), 'optim_d': optim_dF.state_dict(), 'steps': steps,
                    #                     'epoch': epoch})

                generatorM.train()
                generatorF.train()

            steps += 1

            if (steps == 100005):
                exit()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='VCC2018_wavs')
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--load_vocoder', default='True')
    parser.add_argument('--vocoder_load_gen_path', default='')
    parser.add_argument('--vocoder_load_disc_path', default='')
    parser.add_argument('--input_training_fileM', default='')
    parser.add_argument('--input_training_fileF', default='')
    parser.add_argument('--input_validation_fileM', default='')
    parser.add_argument('--input_validation_fileF', default='')
    parser.add_argument('--checkpoint_path', default='')
    parser.add_argument('--config', default='config_v1.json')
    parser.add_argument('--training_epochs', default=1253950, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=100, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=100, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--cycle_loss_lambda', default=45, type=int)
    parser.add_argument('--identity_loss_due', default=40000, type=int)
    parser.add_argument('--identity_loss_lambda', default=20, type=int)
    parser.add_argument('--fm_loss_lambda', default=1, type=int)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    h.batch_size = 8

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
