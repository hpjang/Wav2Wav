from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator
import librosa
from librosa.util import normalize
import numpy as np

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    with open('config_v1.json') as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    generatorA2B = Generator(h).to(device)
    generatorB2A = Generator(h).to(device)

    load_path = a.checkpoint_file.split('/')

    checkpointA2B = torch.load(os.path.join(load_path[0], 'gF'+load_path[1]), map_location=device)
    checkpointB2A = torch.load(os.path.join(load_path[0], 'gM'+load_path[1]), map_location=device)
    generatorA2B.load_state_dict(checkpointA2B['generator'])
    generatorB2A.load_state_dict(checkpointB2A['generator'])


    filelist_S = os.listdir(a.source_files)
    filelist_T = os.listdir(a.target_files)

    generatorA2B.eval()
    generatorB2A.eval()
    generatorA2B.remove_weight_norm()
    generatorB2A.remove_weight_norm()
    with torch.no_grad():
        scores = []
        for i, (filename_S, filename_T) in enumerate(zip(filelist_S, filelist_T)):
            wav, sr = load_wav(os.path.join(a.source_files, filename_S))
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            
            real_A_mel = get_mel(wav.unsqueeze(0))
            aa,bb,cc = real_A_mel.shape
            fake_B = generatorA2B(wav.unsqueeze(0).unsqueeze(0))
            # fake_B_mel = get_mel(fake_B.squeeze(0))[:aa,:bb,:cc]
            cycle_A = generatorB2A(fake_B)
            cycle_A_mel = get_mel(cycle_A.squeeze(0))[:aa,:bb,:cc]
            cycleLoss = torch.mean(torch.abs(real_A_mel - cycle_A_mel))
            scores.append([cycleLoss.item()])


            wav, sr = load_wav(os.path.join(a.target_files, filename_T))
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            
            real_A_mel = get_mel(wav.unsqueeze(0))
            aa,bb,cc = real_A_mel.shape
            fake_B = generatorB2A(wav.unsqueeze(0).unsqueeze(0))
            # fake_B_mel = get_mel(fake_B.squeeze(0))[:aa,:bb,:cc]
            cycle_A = generatorA2B(fake_B)
            cycle_A_mel = get_mel(cycle_A.squeeze(0))[:aa,:bb,:cc]
            cycleLoss = torch.mean(torch.abs(real_A_mel - cycle_A_mel))

        print("loss:    ",np.mean(scores))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_files', required=True)
    parser.add_argument('--target_files', required=True)
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join('config_v1.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

