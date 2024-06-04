set -x

CHECKPOINT_HOME=
LLAMA2_13B=
IWSLT_DATA_HOME=

# total batch size 128
# ct.ours total batch size 8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
if_ours=False
learning_rate=2e-5
gradient_accumulation_steps=16 
save_steps=2000
num_train_epochs=3
model_name=${1:-"llama-13b.mt12000"}
extra_args=""
if [ $model_name == "llama-13b.mt12000" ];then 
    model_path=$LLAMA2_13B
    ft_data=$IWSLT_DATA_HOME/mt12000.json
elif [ $model_name == "llama-13b.mt12000.ct.ours" ];then  
    model_path=${CHECKPOINT_HOME}/llama-13b.mt12000
    ft_data=$IWSLT_DATA_HOME/mt12000.json
    if_ours=True
    learning_rate=2e-6
    gradient_accumulation_steps=1
    save_steps=100
    num_train_epochs=1 
    extra_args="--negative_instruction True --alpha 0.05 --max_steps 100 " 
elif [ $model_name == "llama-13b.alpaca_en.mt12000" ];then 
    model_path=${LLAMA2_13B}
    ft_data=${IWSLT_DATA_HOME}/alpaca_en.mt12000.json
elif [ $model_name == "llama-13b.alpaca_en.mt12000.ct.ours" ];then  
    model_path=${CHECKPOINT_HOME}/llama-13b.alpaca_en.mt12000
    ft_data=${IWSLT_DATA_HOME}/mt12000.json
    if_ours=True
    save_steps=100
    learning_rate=2e-6
    gradient_accumulation_steps=1
    num_train_epochs=1 
    extra_args="--negative_instruction True --max_steps 100 "
fi


output_dir=${CHECKPOINT_HOME}/${model_name}
mkdir -p ${output_dir}

torchrun --nproc_per_node=8 --master_port=5456 train.py \
    --model_name_or_path ${model_path} \
    --data_path ${ft_data} \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps ${save_steps} \
    --learning_rate ${learning_rate} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap offload" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --ours ${if_ours} ${extra_args}| tee $output_dir/train.log
