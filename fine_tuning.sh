set -x
CHECKPOINT_HOME=
LLAMA2_7B=
LLAMA3_8B=
IWSLT_DATA_HOME=
WMT_DATA_HOME=


# total batch size 128
# ct.ours total batch size 8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
if_ours=False
learning_rate=2e-5
gradient_accumulation_steps=16
save_steps=1000
num_train_epochs=3
task_name=${1:-"llama-7b.mt12000"}
extra_args=""

if [ $task_name == "llama3-8b.mt12000" ];then 
    model_path=${LLAMA3_8B}
    ft_data=${IWSLT_DATA_HOME}/mt12000.json
elif [ $task_name == "llama3-8b.mt12000.ct.ours" ];then 
    model_path=${CHECKPOINT_HOME}/llama3-8b.mt12000
    ft_data=${IWSLT_DATA_HOME}/mt12000.json
    if_ours=True
    learning_rate=2e-6
    gradient_accumulation_steps=1
    save_steps=20
    num_train_epochs=1 
    extra_args="--negative_instruction True --max_steps 100 " 
elif [ $task_name == "llama3-8b.mt12000.post_ins" ];then 
    model_path=${LLAMA3_8B}
    ft_data=${IWSLT_DATA_HOME}/mt12000.json
    extra_args="--post_instruction True " 
elif [ $task_name == "llama-7b.mt12000" ];then 
    model_path=${LLAMA2_7B}
    ft_data=${IWSLT_DATA_HOME}/mt12000.json
elif [ $task_name == "llama-7b.mt24000" ];then 
    model_path=${LLAMA2_7B}
    ft_data=${IWSLT_DATA_HOME}/mt24000.json
elif [ $task_name == "llama-7b.mt48000" ];then 
    model_path=${LLAMA2_7B}
    ft_data=${IWSLT_DATA_HOME}/mt48000.json
elif [ $task_name == "llama-7b.mt96000" ];then 
    model_path=${LLAMA2_7B}
    ft_data=${IWSLT_DATA_HOME}/mt96000.json
elif [ $task_name == "llama-7b.wmt-3" ];then 
    model_path=${LLAMA2_7B}
    ft_data=${WMT_DATA_HOME}/wmt-3.51k.alpaca_format.json
    num_train_epochs=1
elif [ $task_name == "llama3-8b.wmt-3" ];then 
    model_path=${LLAMA3_8B}
    ft_data=${WMT_DATA_HOME}/wmt-3.51k.alpaca_format.json
    num_train_epochs=1
elif [ $task_name == "llama3-8b.wmt-3.ct.ours" ];then 
    model_path=${CHECKPOINT_HOME}/llama3-8b.wmt-3
    ft_data=${WMT_DATA_HOME}/wmt-3.51k.alpaca_format.json
    if_ours=True
    save_steps=100
    learning_rate=2e-6
    gradient_accumulation_steps=1
    num_train_epochs=1 
    extra_args="--negative_instruction True --max_steps 100 "
elif [ $task_name == "llama3-8b.wmt-3.post_ins" ];then 
    model_path=${LLAMA3_8B}
    ft_data=${WMT_DATA_HOME}/wmt-3.51k.alpaca_format.json
    num_train_epochs=1
    extra_args="--post_instruction True " 
elif [ $task_name == "llama-7b.wmt-3.ct.ours" ];then 
    model_path=${CHECKPOINT_HOME}/llama-7b.wmt-3
    ft_data=${WMT_DATA_HOME}/wmt-3.51k.alpaca_format.json
    if_ours=True
    save_steps=100
    learning_rate=2e-6
    gradient_accumulation_steps=1
    num_train_epochs=1 
    extra_args="--negative_instruction True --max_steps 100 "
elif [ $task_name == "llama-7b.alpaca_en.wmt-3" ];then 
    model_path=${LLAMA2_7B}
    ft_data=${WMT_DATA_HOME}/alpaca_en.wmt-3.json
    num_train_epochs=1
elif [ $task_name == "llama-7b.alpaca_en.wmt-3.ct.ours" ];then
    model_path=${CHECKPOINT_HOME}/llama-7b.alpaca_en.wmt-3
    ft_data=${WMT_DATA_HOME}/alpaca_en.wmt-3.json
    if_ours=True
    save_steps=50
    learning_rate=2e-6
    gradient_accumulation_steps=1
    num_train_epochs=1 
    extra_args="--negative_instruction True --max_steps 100 "
elif [ $task_name == "llama-7b.mt24000.ct.ours" ];then 
    model_path=${CHECKPOINT_HOME}/llama-7b.mt24000
    ft_data=${IWSLT_DATA_HOME}/mt24000.json
    if_ours=True
    learning_rate=2e-6
    gradient_accumulation_steps=1
    save_steps=100
    num_train_epochs=1 
    extra_args="--negative_instruction True --max_steps 100 " 
elif [ $task_name == "llama-7b.mt48000.ct.ours" ];then 
    model_path=${CHECKPOINT_HOME}/llama-7b.mt48000
    ft_data=${IWSLT_DATA_HOME}/mt48000.json
    if_ours=True
    learning_rate=2e-6
    gradient_accumulation_steps=1
    save_steps=100
    num_train_epochs=1 
    extra_args="--negative_instruction True --max_steps 100 " 
elif [ $task_name == "llama-7b.mt96000.ct.ours" ];then
    model_path=${CHECKPOINT_HOME}/llama-7b.mt96000
    ft_data=${IWSLT_DATA_HOME}/mt96000.json
    if_ours=True
    learning_rate=2e-6
    gradient_accumulation_steps=1
    save_steps=100
    num_train_epochs=1 
    extra_args="--negative_instruction True --max_steps 100 " 
elif [ $task_name == "llama-7b.mt12000.ct.ours" ];then 
    model_path=${CHECKPOINT_HOME}/llama-7b.mt12000
    ft_data=${IWSLT_DATA_HOME}/mt12000.json
    if_ours=True
    learning_rate=2e-6
    gradient_accumulation_steps=1
    save_steps=20
    num_train_epochs=1 
    extra_args="--negative_instruction True --max_steps 100 " 
elif [ $task_name == "llama-7b.mt12000.ct.ours.alpha" ];then  
    # 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.30 0.50 0.70 0.90 1.00
    model_path=${CHECKPOINT_HOME}/llama-7b.mt12000
    ft_data=${IWSLT_DATA_HOME}/mt12000.json
    if_ours=True
    learning_rate=2e-6
    gradient_accumulation_steps=1
    save_steps=100
    num_train_epochs=1 
    alpha=${2} # for analysis experiments
    extra_args="--negative_instruction True --max_steps 100 --alpha ${alpha} " 
    task_name=llama-7b.mt12000.ct.ours.alpha${alpha}
elif [ $task_name == "llama-7b.mt12000.post_ins" ];then 
    model_path=${LLAMA2_7B}
    ft_data=${IWSLT_DATA_HOME}/mt12000.json
    extra_args="--post_instruction True " 
elif [ $task_name == "alma-7b.alma_data.ct.ours" ];then 
    model_path=$ALMA_7B
    ft_data=$ALMA_DATA_HOME/alma_alpaca_format.json
    if_ours=True
    learning_rate=2e-6
    gradient_accumulation_steps=1 # 8GPU
    save_steps=100
    num_train_epochs=1 
    extra_args="--negative_instruction True --max_steps 100 --alma_template True " 
elif [ $task_name == "llama-7b.wmt-3.post_ins" ];then 
    model_path=${LLAMA2_7B}
    ft_data=${WMT_DATA_HOME}/wmt-3.51k.alpaca_format.json
    num_train_epochs=1
    extra_args="--post_instruction True " 
elif [ $task_name == "llama-7b.alpaca_en" ];then 
    model_path=${LLAMA2_7B}
    ft_data=${IWSLT_DATA_HOME}/alpaca_en.json
elif [ $task_name == "llama-7b.alpaca_en.mt12000" ];then 
    model_path=${LLAMA2_7B}
    ft_data=${IWSLT_DATA_HOME}/alpaca_en.mt12000.json
elif [ $task_name == "llama-7b.alpaca_en.mt12000.ct.ours" ];then  
    model_path=${CHECKPOINT_HOME}/llama-7b.alpaca_en.mt12000
    ft_data=${IWSLT_DATA_HOME}/mt12000.json
    if_ours=True
    save_steps=100
    learning_rate=2e-6
    gradient_accumulation_steps=1
    num_train_epochs=1 
    extra_args="--negative_instruction True --max_steps 100 "
fi
output_dir=${CHECKPOINT_HOME}/${task_name}
mkdir -p $output_dir


torchrun --nproc_per_node=8 --master_port=5458 train.py \
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
    --ours ${if_ours} ${extra_args}| tee $output_dir/train.log 2>&1
