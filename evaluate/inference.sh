set -x 
# inference 
task_name=${1:-"llama-7b.mt12000"}
data=${2:-"iwslt4"}
post_ins=${3:-"False"}
PTL=${4:-"False"}
few_shot=${5:-"False"}
n_shot=${6:-"0"}

CHECKPOINT_HOME=
data_path=

model=${CHECKPOINT_HOME}/${task_name}

start_id=${8:-0}
end_id=${9:-7}
gpu_id=$start_id
    
if [ $data == "iwslt4" ];then
    data_path=${data_path}/iwslt4/test
    output=output
    extra_args=""

    if [ ${post_ins} == "post_ins" ];then
        extra_args="--post-ins True "
        output=output_post_ins
    fi 
    if [ ${few_shot} == "few_shot" ];then
        extra_args="--n ${n_shot} "
        output=output_${n_shot}_shot
    fi 
    lps=${7:-"en-it it-en en-nl nl-en en-ro ro-en it-nl nl-it it-ro ro-it nl-ro ro-nl"}
    for pair in ${lps[@]};do 
        langs=(${pair//-/ })
        src=${langs[0]}
        tgt=${langs[1]} 

        if [ ${PTL} == "PTL" ];then
            extra_args="-pl ${tgt} "
            output=output_PTL
        fi 

        if [ $gpu_id == $end_id ];then
            wait
            gpu_id=$start_id
        fi
        mkdir -p ${model}/$output
        echo "${src}-${tgt}"
        CUDA_VISIBLE_DEVICES=$gpu_id python3 inference_w_vllm.py --model-name-or-path ${model} \
            -lp ${src}-${tgt} \
            -t 0.1 \
            -sa 'beam' \
            -i ${data_path}/tst2017${src}-${tgt}.${src} \
            -o ${model}/$output/${src}-${tgt}.${tgt} ${extra_args}&
        let gpu_id=$gpu_id+1
    done 
elif [ $data == "wmt3" ];then 
    data_path=${data_path}/wmt22/sources
    output=output
    extra_args=""

    if [ ${post_ins} == "post_ins" ];then
        extra_args="--post-ins True "
        output=output_post_ins
    fi 
    if [ ${few_shot} == "few_shot" ];then
        extra_args="--n ${n_shot} "
        output=output_${n_shot}_shot
    fi 

    for pair in en-de de-en en-zh zh-en en-cs cs-en en-ja ja-en en-ru ru-en en-uk uk-en fr-de de-fr;do
        langs=(${pair//-/ })
        src=${langs[0]}
        tgt=${langs[1]}

        if [ ${PTL} == "PTL" ];then
            extra_args="-pl ${tgt} "
            output=output_PTL
        fi 

        if [ $gpu_id == $end_id ];then
            wait
            gpu_id=$start_id
        fi
        mkdir -p ${model}/$output


        if [ $gpu_id == $end_id ];then
            wait
            gpu_id=$start_id
        fi

        echo "${src}-${tgt}"

        CUDA_VISIBLE_DEVICES=$gpu_id python3 inference_w_vllm.py --model-name-or-path ${model} \
            -lp ${src}-${tgt} \
            -t 0.1 \
            -sa 'beam' \
            -i ${data_path}/generaltest2022.${src}-${tgt}.src.${src} \
            -o ${model}/${output}/${src}-${tgt}.${tgt} ${extra_args}&

        let gpu_id=$gpu_id+1
    done 
fi
wait 

