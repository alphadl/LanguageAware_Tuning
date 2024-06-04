from collections import OrderedDict
checkpoints="CHECKPOINT_HOME/ours_models"

exp_iwslt4_llama2_checkpoint_list_ = {
    "LLaMA-7B": f"{checkpoints}/llama-7b",
    "mt-7B": f"{checkpoints}/llama-7b.mt12000",
    "mt-post_ins": f"{checkpoints}/llama-7b.mt12000.post_ins",
    "mt-ptl": f"{checkpoints}/llama-7b.mt12000", 
    "mt-1_shot": f"{checkpoints}/llama-7b.mt12000", 
    "mt-5_shot": f"{checkpoints}/llama-7b.mt12000", 
    "contradecode": f"{checkpoints}/llama-7b.mt12000", 
    "mt-7B+negative instruction": f"{checkpoints}/llama-7b.mt12000.ct.unions/checkpoint-100",
}
exp_iwslt4_llama2_checkpoint_list = OrderedDict(exp_iwslt4_llama2_checkpoint_list_)


exp_iwslt4_llama3_checkpoint_list_ = {
    "LLaMA3-8B": f"{checkpoints}/llama3-8b",
    "llama3-mt-8B": f"{checkpoints}/llama3-8b.mt12000",
    "llama3-mt-post_ins": f"{checkpoints}/llama3-8b.mt12000.post_ins",
    "llama3-mt-ptl": f"{checkpoints}/llama3-8b.mt12000", 
    "llama3-mt-1_shot": f"{checkpoints}/llama3-8b.mt12000", 
    "llama3-mt-5_shot": f"{checkpoints}/llama3-8b.mt12000", 
    "mt-llama3-contradecode": f"{checkpoints}/llama3-8b.mt12000", 
    "llama3-mt-8B+negative instruction": f"{checkpoints}/llama3-8b.mt12000.ct.unions/checkpoint-100",
}
exp_iwslt4_llama3_checkpoint_list = OrderedDict(exp_iwslt4_llama3_checkpoint_list_)

exp_wmt3_llama2_checkpoint_list_ = {
    "LLaMA-7B": f"{checkpoints}/llama-7b",
    "mt-7B": f"{checkpoints}/llama-7b.wmt-3",
    "mt-post_ins": f"{checkpoints}/llama-7b.wmt-3.post_ins",
    "mt-ptl": f"{checkpoints}/llama-7b.wmt-3", 
    "mt-1_shot": f"{checkpoints}/llama-7b.wmt-3", 
    "mt-5_shot": f"{checkpoints}/llama-7b.wmt-3", 
    "contradecode": f"{checkpoints}/llama-7b.wmt-3", 
    "mt-7B+negative instruction": f"{checkpoints}/llama-7b.wmt-3.ct.unions/checkpoint-100",
}
exp_wmt3_llama2_checkpoint_list = OrderedDict(exp_wmt3_llama2_checkpoint_list_)


exp_wmt3_llama3_checkpoint_list_ = {
    "LLaMA3-8B": f"{checkpoints}/llama3-8b",
    "llama3-mt-8B": f"{checkpoints}/llama3-8b.wmt-3",
    "llama3-mt-post_ins": f"{checkpoints}/llama3-8b.wmt-3.post_ins",
    "llama3-mt-ptl": f"{checkpoints}/llama3-8b.wmt-3", 
    "llama3-mt-1_shot": f"{checkpoints}/llama3-8b.wmt-3", 
    "llama3-mt-5_shot": f"{checkpoints}/llama3-8b.wmt-3", 
    "mt-llama3-contradecode": f"{checkpoints}/llama3-8b.wmt-3", 
    "llama3-mt-8B+negative instruction": f"{checkpoints}/llama3-8b.wmt-3.ct.unions/checkpoint-100",
}
exp_wmt3_llama3_checkpoint_list = OrderedDict(exp_wmt3_llama3_checkpoint_list_)

exp_ablation_over_alpha_checkpoint_list_ = {
    "0.0": f"{checkpoints}/llama-7b.mt12000",
    "0.01": f"{checkpoints}/llama-7b.mt12000.ct.unions.alpha0.01",
    "0.02": f"{checkpoints}/llama-7b.mt12000.ct.unions.alpha0.02",
    "0.03": f"{checkpoints}/llama-7b.mt12000.ct.unions.alpha0.03",
    "0.04": f"{checkpoints}/llama-7b.mt12000.ct.unions.alpha0.04",
    "0.05": f"{checkpoints}/llama-7b.mt12000.ct.unions/checkpoint-100",
    "0.06": f"{checkpoints}/llama-7b.mt12000.ct.unions.alpha0.06",
    "0.07": f"{checkpoints}/llama-7b.mt12000.ct.unions.alpha0.07",
    "0.08": f"{checkpoints}/llama-7b.mt12000.ct.unions.alpha0.08",
    "0.09": f"{checkpoints}/llama-7b.mt12000.ct.unions.alpha0.09",
    "0.1": f"{checkpoints}/llama-7b.mt12000.ct.unions.alpha0.10",
    "0.3": f"{checkpoints}/llama-7b.mt12000.ct.unions.alpha0.30",
    "0.5": f"{checkpoints}/llama-7b.mt12000.ct.unions.alpha0.50",
    "0.7": f"{checkpoints}/llama-7b.mt12000.ct.unions.alpha0.70",
    "0.9": f"{checkpoints}/llama-7b.mt12000.ct.unions.alpha0.90",
    "1.0": f"{checkpoints}/llama-7b.mt12000.ct.unions.alpha1.00"
}
exp_ablation_over_alpha_checkpoint_list = OrderedDict(exp_ablation_over_alpha_checkpoint_list_)


exp_step_analysis_checkpoint_list_ = {
    "0": f"{checkpoints}/llama-7b.mt12000",
    "20": f"{checkpoints}/llama-7b.mt12000.ct.unions/checkpoint-20",
    "40": f"{checkpoints}/llama-7b.mt12000.ct.unions/checkpoint-40",
    "60": f"{checkpoints}/llama-7b.mt12000.ct.unions/checkpoint-60",
    "80": f"{checkpoints}/llama-7b.mt12000.ct.unions/checkpoint-80",
    "100": f"{checkpoints}/llama-7b.mt12000.ct.unions/checkpoint-100",
}
exp_step_analysis_checkpoint_list = OrderedDict(exp_step_analysis_checkpoint_list_)


exp_scaling_analysis_checkpoint_list_ = {
    "7B": f"{checkpoints}/llama-7b.mt12000",
    "13B": f"{checkpoints}/llama-13b.mt12000",
    "7B+negative instruction": f"{checkpoints}/llama-7b.mt12000.ct.unions/checkpoint-100",
    "13B+negative instruction": f"{checkpoints}/llama-13b.mt12000.ct.unions/checkpoint-100",
    "12k": f"{checkpoints}/llama-7b.mt12000",
    "24k": f"{checkpoints}/llama-7b.mt24000",
    "48k": f"{checkpoints}/llama-7b.mt48000",
    "96k": f"{checkpoints}/llama-7b.mt96000",
    "12k+negative instruction": f"{checkpoints}/llama-7b.mt12000.ct.unions/checkpoint-100",
    "24k+negative instruction": f"{checkpoints}/llama-7b.mt24000.ct.unions/checkpoint-100",
    "48k+negative instruction": f"{checkpoints}/llama-7b.mt48000.ct.unions/checkpoint-100",
    "96k+negative instruction": f"{checkpoints}/llama-7b.mt96000.ct.unions/checkpoint-100",
}
exp_scaling_analysis_checkpoint_list = OrderedDict(exp_scaling_analysis_checkpoint_list_)


exp_general_task_checkpoint_list_ = {
    "alpaca_7b_mt": f"{checkpoints}/llama-7b.alpaca_en.mt12000",
    "alpaca_7b_mt+negative instruction": f"{checkpoints}/llama-7b.alpaca_en.mt12000.ct.unions/checkpoint-100",
    "alpaca_13b_mt": f"{checkpoints}/llama-13b.alpaca_en.mt12000",
    "alpaca_13b_mt+negative instruction": f"{checkpoints}/llama-13b.alpaca_en.mt12000.ct.unions/checkpoint-100",
}
exp_general_task_checkpoint_list = OrderedDict(exp_general_task_checkpoint_list_)
