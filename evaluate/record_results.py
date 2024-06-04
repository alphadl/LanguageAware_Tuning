import os 
from sacrebleu.metrics import BLEU 
import fasttext 
from tqdm import tqdm 
from checkpoint_list import *

TEST_DATA="data_path"


class LanguageIdentification:
    def __init__(self):
        pretrained_lang_model = "./lid.176.bin"
        self.model = fasttext.load_model(pretrained_lang_model)
    def predict_lang(self, text):
        predictions = self.model.predict(text, k=1) # returns top 1 matching languages
        return predictions
LANGUAGE = LanguageIdentification()

def compute_sacrebleu(tgt, test_file, hyp_file, **kwargs):
    bleu = BLEU(trg_lang = tgt)

    predictions_value = []
    references_value = []
    with open(hyp_file, "r", encoding="utf-8") as hyp:
        for line in hyp:
            predictions_value.append(line.strip())
    with open(test_file, "r", encoding="utf-8") as test:
        for line in test:
            references_value.append(line.strip())
    
    assert len(references_value) == len(predictions_value)
    
    return bleu.corpus_score(predictions_value, [references_value]).score

def compute_otr(src, tgt, hyp_file, **kwargs):
    tgt_index = []
    with open(hyp_file, 'r', encoding='utf-8') as f2:
        total_num=0
        i = 0
        j = 0
        k = 0
        for id, line in enumerate(f2):
            total_num += 1
            language = LANGUAGE.predict_lang(line.strip())
            l = language[0][0].split('_')[-1]
            if l == tgt:
                i+=1
                tgt_index.append(id)
            if l == src:
                j+=1
            if l == "en":
                k+=1
    return {tgt: 100 * i / total_num, src: 100 * j / total_num, "en": 100 * k / total_num}, tgt_index

def results_compute_and_write(checkpoint_list, checkpoint_list_name, dataset_name="iwslt4", write_mode="w"):
    all_results = {}

    for model_name, path in checkpoint_list.items():
        if dataset_name == "wmt3":
            if "contra" in model_name:
                lang_pairs = ["en-cs", "cs-en", "en-ja", "ja-en", "en-ru", "ru-en", "en-uk", "uk-en", "fr-de", "de-fr"]
            else:
                lang_pairs = ["en-de", "de-en", "en-zh", "zh-en", "en-cs", "cs-en", "en-ja", "ja-en", "en-ru", "ru-en", "en-uk", "uk-en", "fr-de", "de-fr"]
            test_data_path = f"{TEST_DATA}/wmt22/references"
            test_src_path = f"{TEST_DATA}/wmt22/sources"
        else: # iwslt4
            if "contra" in model_name:
                lang_pairs = "it-nl nl-it it-ro ro-it nl-ro ro-nl".split()
            else:
                lang_pairs = "en-it it-en en-nl nl-en en-ro ro-en it-nl nl-it it-ro ro-it nl-ro ro-nl".split()
            test_data_path = f"{TEST_DATA}/iwslt4/test"
        print(model_name)
        all_results[model_name] = {}
        all_results[model_name]["sacrebleu"] = {}
        all_results[model_name]["otr"] = {}
        all_results[model_name]["bleurt"] = {}
        
        for pair in tqdm(lang_pairs):
            src, tgt = pair.split("-")
            
            if dataset_name == "iwslt4": 
                if "post_ins" in model_name:
                    output = "output_post_ins"
                elif "ptl" in model_name:
                    output = "output_PTL"
                elif "1_shot" in model_name:
                    output = "output_1_shot"
                elif "5_shot" in model_name:
                    output = "output_5_shot"
                elif "contradecode" in model_name:
                    output = "contradecode_output"
                else:
                    output = "output"
                args = {
                    "test_data_path": test_data_path, 
                    "src": src, 
                    "tgt": tgt, 
                    "model_path": path,
                    "test_file": os.path.join(test_data_path, f"tst2017{tgt}-{src}.{tgt}"),
                    "hyp_file": os.path.join(path, output, f"{src}-{tgt}.{tgt}.hyp") if "contra" not in model_name else os.path.join(path, "contradecode_output", f"{src}-{tgt}/contrastive-None--0.1-lang-en+{src}--0.5.txt"),
                    "src_file": os.path.join(test_data_path, f"tst2017{pair}.{src}")
                    }
            elif dataset_name == "wmt3":
                if src == "cs" or tgt == "cs":
                    test_file = os.path.join(test_data_path, f"generaltest2022.{src}-{tgt}.ref.B.{tgt}")
                else:
                    test_file = os.path.join(test_data_path, f"generaltest2022.{src}-{tgt}.ref.A.{tgt}")
                
                if "post_ins" in model_name:
                    output = "output_post_ins"
                elif "ptl" in model_name:
                    output = "output_PTL"
                elif "1_shot" in model_name:
                    output = "output_1_shot"
                elif "5_shot" in model_name:
                    output = "output_5_shot"
                elif "contradecode" in model_name:
                    output = "contradecode_output"
                else:
                    output = "output"
                args = {
                    "test_data_path": test_data_path, 
                    "src": src, 
                    "tgt": tgt, 
                    "model_path": path, 
                    "test_file": test_file, 
                    "hyp_file": os.path.join(path, output, f"{src}-{tgt}.{tgt}.hyp") if "contra" not in model_name else os.path.join(path, "contradecode_output", f"{src}-{tgt}/contrastive-None--0.1-lang-en+{src}--0.5.txt"),
                    "src_file": os.path.join(test_src_path, f"generaltest2022.{src}-{tgt}.src.{src}") 
                    }
                    
            all_results[model_name]["sacrebleu"][f"{src}-{tgt}"] = \
                    compute_sacrebleu(**args)
            result, tgt_index = compute_otr(**args)
            all_results[model_name]["otr"][f"{src}-{tgt}"] = \
                100 - result[tgt]

    def get_str(head_str_, model_name, all_results, result_key, direction):
        if dataset_name == "wmt3":
            if direction == "supervised":
                if not "en-de" in all_results[model_name][result_key].keys():
                    return ""  
                for lang_pair in ["en-de", "de-en", "en-zh", "zh-en"]:
                    head_str_ = head_str_ + str(all_results[model_name][result_key][lang_pair]) + "\t"
            elif direction == "zero-shot":
                for lang_pair in ["en-cs", "cs-en", "en-ja", "ja-en", "en-ru", "ru-en", "en-uk", "uk-en", "fr-de", "de-fr"]:
                    head_str_ = head_str_ + str(all_results[model_name][result_key][lang_pair]) + "\t"
        else: # iwslt 
            if direction == "supervised":
                if not "en-it" in all_results[model_name][result_key].keys():
                    return ""  
                for lang_pair in "en-it it-en en-nl nl-en en-ro ro-en".split():
                    head_str_ = head_str_ + str(all_results[model_name][result_key][lang_pair]) + "\t"
            elif direction == "zero-shot":
                for lang_pair in "it-nl nl-it it-ro ro-it nl-ro ro-nl".split():
                    head_str_ = head_str_ + str(all_results[model_name][result_key][lang_pair]) + "\t"
        return head_str_

    print(all_results)
    print(f"output score: ./results/record_results.scores")
    output_file = open(f"./results/record_results.scores", write_mode)

    output_file.write(f">> {checkpoint_list_name}\n")
    output_file.write("supervised translation bleu\n")
    for model_name in all_results.keys():
        ourput_str = model_name + "\t"
        ourput_str = get_str(ourput_str, model_name, all_results, "sacrebleu", "supervised")
        output_file.write(ourput_str + "\n")
    output_file.write("\n\n")

    output_file.write("zero-shot translation bleu\n")
    for model_name in all_results.keys():
        ourput_str = model_name + "\t"
        ourput_str = get_str(ourput_str, model_name, all_results, "sacrebleu", "zero-shot")
        output_file.write(ourput_str + "\n")
    output_file.write("\n\n")

    output_file.write("supervised translation otr\n")
    for model_name in all_results.keys():
        ourput_str = model_name + "\t"
        ourput_str = get_str(ourput_str, model_name, all_results, "otr", "supervised")
        output_file.write(ourput_str + "\n")
    output_file.write("\n\n")

    output_file.write("zero-shot translation otr\n")
    for model_name in all_results.keys():
        ourput_str = model_name + "\t"
        ourput_str = get_str(ourput_str, model_name, all_results, "otr", "zero-shot")
        output_file.write(ourput_str + "\n")
    output_file.write("\n\n")



def main():
    results_compute_and_write(exp_iwslt4_llama2_checkpoint_list, checkpoint_list_name="exp_iwslt4_llama2_checkpoint_list")
    results_compute_and_write(exp_iwslt4_llama3_checkpoint_list, checkpoint_list_name="exp_iwslt4_llama3_checkpoint_list", write_mode="a")
    results_compute_and_write(exp_wmt3_llama2_checkpoint_list, checkpoint_list_name="exp_wmt3_llama2_checkpoint_list", dataset_name="wmt3", write_mode="a")
    results_compute_and_write(exp_wmt3_llama3_checkpoint_list, checkpoint_list_name="exp_wmt3_llama3_checkpoint_list", dataset_name="wmt3", write_mode="a")
    results_compute_and_write(exp_ablation_over_alpha_checkpoint_list, checkpoint_list_name="exp_ablation_over_alpha_checkpoint_list", write_mode="a")
    results_compute_and_write(exp_step_analysis_checkpoint_list, checkpoint_list_name="exp_step_analysis_checkpoint_list", write_mode="a")
    results_compute_and_write(exp_scaling_analysis_checkpoint_list, checkpoint_list_name="exp_scaling_analysis_checkpoint_list",write_mode="a")
    results_compute_and_write(exp_general_task_checkpoint_list, checkpoint_list_name="exp_general_task_checkpoint_list", write_mode="a")

if __name__ == "__main__":
    main()

