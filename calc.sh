CHK=$1
SF=$2
TF=$3
for cp in `ls ${CHK} | grep "gF"`
do
    if [[ ${cp::2} == "gF" ]]
    then
        # echo ${cp:2}
    	python3 calc_.py  --source_files ${SF} --target_files ${TF}  --checkpoint_file $CHK/${cp:2} 
    fi
done