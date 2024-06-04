set -x
exp_name=${1:-"reproduce"}
start_id=0
end_id=8
gpu_id=$start_id

CHECKPOINT_HOME=
data_path=


cd ContraDecode
if [ ${exp_name} == "reproduce" ] ;then
    # reproduce
    model=PATH/Llama-2-7b-chat-hf
    flores_200_home=${data_path}/flores-200
    src_path=${flores_200_home}/devtest/eng_Latn.devtest
    tgt_path=${flores_200_home}/devtest/deu_Latn.devtest

    python -m scripts.run \
        --model_path ${model} \
        --language_pairs en-de \
        --source_contrastive 1 --source_weight -0.0 \
        --language_contrastive en \
        --language_weight -0.5 \
        --src_path ${src_path} \
        --tgt_path ${tgt_path} 

    pushd out/flores/en-de
        sacrebleu ref.txt < contrastive-1--0.0-lang-en--0.5.txt --metrics chrf 
        sacrebleu ref.txt < direct.txt --metrics chrf 
    popd 
elif [ ${exp_name} == "llama-7b.mt12000" ] || [ ${exp_name} == "llama3-8b.mt12000" ] ;then
    data_path=${data_path}/iwslt4/test
    model=${CHECKPOINT_HOME}/$exp_name
    for pair in it-nl nl-it it-ro ro-it ro-nl nl-ro ;do 
        langs=(${pair//-/ })
        src=${langs[0]}
        tgt=${langs[1]}
        if [ $gpu_id == $end_id ] ;then
            wait
            gpu_id=$start_id
        fi
        src_path=${data_path}/tst2017${src}-${tgt}.${src}
        tgt_path=${data_path}/tst2017${tgt}-${src}.${tgt}
        if [ ! -d "${model}/contradecode_output/${src}-${tgt}/contrastive-1--0.0-lang-en+${src}--0.5" ];then 
            mkdir -p ${model}/contradecode_output/${src}-${tgt}
            CUDA_VISIBLE_DEVICES=$gpu_id python -m scripts.run \
                --model_path ${model} \
                --language_pairs ${src}-${tgt} \
                --language_contrastive en ${src} \
                --language_weight -0.5 \
                --src_path ${src_path} \
                --tgt_path ${tgt_path} \
                --output_path ${model}/contradecode_output/${src}-${tgt} & 
                let gpu_id=$gpu_id+1
                #                 --source_contrastive 1 --source_weight -0.0 \
        fi
    done
elif [ ${exp_name} == "llama-7b.wmt-3" ] || [ ${exp_name} == "llama3-8b.wmt-3" ] ;then
    WMT_HOME=${data_path}/wmt22
    mkdir -p ${model}/contradecode_output
    model=${CHECKPOINT_HOME}/$exp_name
    for pair in en-cs cs-en en-ja ja-en en-ru ru-en en-uk uk-en fr-de de-fr;do
        langs=(${pair//-/ })
        src=${langs[0]}
        tgt=${langs[1]}

        if [ $src == $tgt ];then
            continue
        fi
        if [ $gpu_id == $end_id ];then
            wait
            gpu_id=$start_id
        fi

        echo "${src}-${tgt}"
        if [ ! -f "${model}/contradecode_output/${src}-${tgt}/contrastive-1--0.0-lang-en+${src}--0.5" ];then
            mkdir -p ${model}/contradecode_output/${src}-${tgt}

            src_path=${WMT_HOME}/sources/generaltest2022.${src}-${tgt}.src.${src}
            tgt_path=${WMT_HOME}/sources/generaltest2022.${src}-${tgt}.src.${src} # do not use
            CUDA_VISIBLE_DEVICES=$gpu_id python -m scripts.run \
                --model_path ${model} \
                --language_pairs ${src}-${tgt} \
                --language_contrastive en ${src} \
                --language_weight -0.5 \
                --src_path ${src_path} \
                --tgt_path ${tgt_path} \
                --output_path ${model}/contradecode_output/${src}-${tgt} &
                let gpu_id=$gpu_id+1
                # --source_contrastive 1 --source_weight -0.0 \
        fi
        
    done
fi

wait 