#CHK=cp_hifigan10_step
CHK=$1
OUT=$2
S=$3
T=$4
mkdir -p $OUT/ST
mkdir -p $OUT/TS

for cp in `ls -r ${CHK}`
do
    if [[ ${cp::2} == "gF" ]]
    then
        python3 inference.py --checkpoint_file $CHK/${cp} --input_wavs_dir $S --output_dir $OUT/ST/${cp:3}
        python3 inference.py --checkpoint_file $CHK/gM${cp:2} --input_wavs_dir $T --output_dir $OUT/TS/${cp:3}
    fi
    # if [[ ${cp::2} == "gM" ]]
    # then
    #     python3 inference.py --checkpoint_file $CHK/${cp} --input_wavs_dir $T --output_dir $OUT/TS/${cp:3}
    # fi
done



# for g in `ls -r $CHK`;
# do
# 	if [ ${g:0:2} == 'gF' ]; then
# 		python3 inference.py --checkpoint_file $CHK/${g} --input_wavs_dir 1spkr_SM3 --output_dir $OUT/${g:2}
# 		echo ${g}
# 	fi
# done

# for g in `ls -r $CHK`;
# do
# 	if [ ${g:0:2} == 'gM' ]; then
# 		python3 inference.py --checkpoint_file $CHK/${g} --input_wavs_dir 1spkr_TF1 --output_dir $OUT/${g:2}
# 		echo ${g}
# 	fi
# done
