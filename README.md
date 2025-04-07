# Wav2wav: Wave-to-Wave Voice Conversion

## 📖 Paper Information
- **Title:** Wav2wav: Wave-to-Wave Voice Conversion
- **Author(s):** Changhyeon Jeong, Hyung-pil Chang, In-Chul Yoo, Dongsuk Yook
- **Published In:** Applied Sciences (MDPI)
- **Publication Year:** 2024
- **DOI (Digital Object Identifier):** [https://www.mdpi.com/2076-3417/14/10/4251](https://www.mdpi.com/2076-3417/14/10/4251)

## 📌 Introduction
This repository provides the source code associated with the paper "Wav2wav: Wave-to-Wave Voice Conversion." The purpose of this project is to implement a novel voice conversion architecture that integrates the feature extractor, feature converter, and vocoder into a single module trained in an end-to-end manner.

## 📁 Directory Structure
- `train.py`, `models.py`: The core files of the project. These files have been most frequently modified for training and evaluation purposes.
- `cp_*` directories: Contain trained weights and TensorBoard logs generated during training.
- `gen_wavs*` directories: Store generated audio samples using trained weights.
- `train_*.sh`: Shell scripts used to train `train.py` and its variants. Various hyperparameters are defined within these scripts or in `config_v1.json`.
- `conv.sh`: A script used to generate audio samples from trained weights.(using cp* directories)
  - Usage: `conv.sh <checkpoint_directory> <output_directory> <source_speaker_directory> <target_speaker_directory>`  
  - Example: `conv.sh cp_hifigan_FM gen_wavs_FM 1spkr_SF3 1spkr_TM1`
- `train_mod_loss.py`: A modified version of training where Fourier transform is replaced by the use of prenet across all relevant parts.
- `train_mod_loss2.py`: A further modification where only the first convolutional layer of the prenet is used (log-magnitude spectrogram format) for id_loss and cycle_loss training.
- Final hyperparameter setting: **45_30_0.5**

## 🐳 Docker Setup
To run this project in a Docker container with GPU support:
```bash
docker run --gpus device=0 -it --memory=16G --memory-swap=32G --shm-size=8G --rm -v /your storage path/:/shared_dir/  /your docker images pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
```

### 🔍 Install Required Packages
```bash
apt-get update && apt-get install libsndfile1-dev
pip install librosa==0.8.1 torchaudio==0.9.0 matplotlib tensorboard
```

## 🔍 Usage
1. Clone this repository:
```bash
 git clone https://github.com/hpjang/Wav2Wav.git
```
2. Download VCC2018 dataset:
https://datashare.ed.ac.uk/handle/10283/3061
4. Train the model:
```bash
 bash train_*.sh
```

### ✅ Generate Audio Samples (Wav2wav, Spec2wav)
```bash
./conv.sh "<checkpoint_directory>" "<output_directory>" "<source_speaker_directory>" "<target_speaker_directory>"
```
Example:
```bash
./conv.sh cp_hifigan_FM_45_30_0.5 generated 1spkr_SF3 1spkr_TM1
```

### ✅ Generate Audio Samples (Wav2spec, Spec2spec)
```bash
./conv??.sh "<checkpoint_directory>" "<output_directory>" "<source_speaker_directory>" "<target_speaker_directory>" "<vocoder_type>"
```
Example:
```bash
./convFM.sh cp_hifigan_FM_45_30_0.5 generated 1spkr_SF3 1spkr_TM1 mel
```

## 📂 Pre-Trained Weights
Trained weights can be downloaded from the following URL:
- https://drive.google.com/drive/folders/1vcRphGAObQN57mcI0PME2dCztuG5Imn3?usp=sharing
(Enter the URL for the trained weights here doawnload it and move to cp_* directroy)



## 📜 License
This project is licensed under the [License Name (MIT or Apache License 2.0)]. For more details, see the [LICENSE](./LICENSE) file.

## 📢 Citation
If you use this project, please cite the paper as follows:
```
@article{Jeong2024Wav2wav,
  title={Wav2wav: Wave-to-Wave Voice Conversion},
  author={Changhyeon Jeong and Hyung-pil Chang and In-Chul Yoo and Dongsuk Yook},
  journal={Applied Sciences},
  volume={14},
  number={10},
  pages={4251},
  year={2024},
  doi={https://www.mdpi.com/2076-3417/14/10/4251}
}
```

## 📧 Contact
For questions or requests, please contact: [hpchang@korea.ac.kr]
