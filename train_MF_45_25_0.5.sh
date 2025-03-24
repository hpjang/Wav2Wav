# Sample training script to convert between VCC2SF3 and VCC2TF1
# Continues training from epoch 500

python3 -W ignore::UserWarning -m train \
    --vocoder_load_gen_path /shared_dir/HiFi-GAN_VCC2018/hifi-gan/cp_hifigan_2spkr_SM3TF1_all/g_00100000 \
    --vocoder_load_disc_path /shared_dir/HiFi-GAN_VCC2018/hifi-gan/cp_hifigan_2spkr_SM3TF1_all/do_00100000 \
    --input_training_fileM VCC_meta2/original/SM3/training.txt \
    --input_training_fileF VCC_meta2/original/TF1/training.txt \
    --input_validation_fileM VCC_meta2/original/SM3/test.txt \
    --input_validation_fileF VCC_meta2/original/TF1/test.txt \
    --checkpoint_path cp_hifigan_MF_45_25_0.5 \
    --cycle_loss_lambda 45 \
    --identity_loss_lambda 25 \
    --identity_loss_due 100001 \
    --fm_loss_lambda 0.5 \
    # --checkpoint_interval 95400 \
    # --summary_interval 10000 \
    # --validation_interval 10000 \