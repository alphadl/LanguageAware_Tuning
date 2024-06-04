import json
import argparse
from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
import torch
import random
import numpy as np
from vllm import LLM, SamplingParams


# Instruction language, default: 'en'
lang_instruction = {
    'de': {'de': "Deutsch", 'en': "Englisch", 'ja': "Japanisch", 'zh': "Chinesisch", "ko": "Koreanisch", "cs": "Tschechisch", "fr": "Französisch", "uk": "ukrainisch", "ru":"Russisch" },
    'en': {'de': "German", 'en': "English", 'ja': "Japanese", 'zh': "Chinese", "ko": "Korean", "cs": "Czech", "fr": "French", "uk": "Ukrainian", "ru": "Russian", 'ro': "Romanian", 'it': "Italian", "nl": "Dutch"},
    'ja': {'de': "ドイツ語", 'en': "英語", 'ja': "日本語", 'zh': "中国語", "ko": "韓国語", "cs": "チェコ語", "fr": "フランス語", "uk": "ウクライナ語", "ru":"ロシア" },
    'zh': {'de': "德语", 'en': "英语", 'ja': "日语", 'zh': "中文", "ko": "韩语", "cs": "捷克语", "fr": "法语", "uk": "乌克兰语", "ru":"俄语" },
    "ko": {'de': "독일 사람", 'en': "영어", 'ko': "한국인", 'zh': "중국인", "ja": "일본어", "cs": "체코 사람", "fr": "프랑스 국민", "uk": "우크라이나 인", "ru":"러시아인" }, 
    "cs": {'de': "Němec", 'en': "Angličtina", 'ko': "korejština", 'zh': "čínština", "ja": "japonský", "cs": "čeština", "fr": "francouzština", "uk": "ukrajinština", "ru":"ruština" }, 
    "fr": {'de': "Allemand", 'en': "Anglais", 'ko': "coréen", 'zh': "Chinois", "ja": "Japonais", "cs": "tchèque", "fr": "Français", "uk": "ukrainien", "ru":"russe" }, 
    "uk": {'de': "Німецький", 'en': "англійська", 'ko': "корейська", 'zh': "китайський", "ja": "Японський", "cs": "чеська", "fr": "французька", "uk": "українська", "ru":"російський" }, 
    "ru": {'de': "Немецкий", 'en': "Английский", 'ko': "Корейский", 'zh': "Китайский", "ja": "Японский", "cs": "Чешский", "fr": "Французский", "uk": "украинец", "ru":"Русский" }, 
    "ro": {"en": "Engleză", "ro": "Română", "it": "Italiană", "nl": "olandeză"}, 
    "it": {"en": "Inglese", "ro": "rumeno", "it": "Italiano", "nl": "Olandese"}, 
    "nl": {"en": "Engels", "ro": "Roemeense", "it": "Italiaans", "nl": "Nederlands"}, 
}

id2name = {"zho_Hans": "Chinese (Simplified)", "eng_Latn": "English", 
           "deu_Latn": "German", "hin_Deva": "Hindi", 
           "arb_Arab": "Modern Standard Arabic", "kor_Hang": "Korean"}


# Special tokens in llama
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

POST_INS_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Input:\n{input}\n\n### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
multilingual_PROMPT_DICT = {
                            "en":{
                    "prompt_input": (
                        "Below is an instruction that describes a task, paired with an input that provides further context. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
                    ),
                    "prompt_no_input": (
                        "Below is an instruction that describes a task. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\n{instruction}\n\n### Response:"
                    ),
                },
                            "zh":{
                    "prompt_input": (
                        "下面是描述任务的指令，并与提供进一步上下文的输入配对。 "
                        "编写适当完成请求的响应。\n\n"
                        "### 指令:\n{instruction}\n\n### 输入:\n{input}\n\n### 回复:"
                    ),
                },
                            "ko":{
                    "prompt_input": (
                        "다음은 추가 컨텍스트를 제공하는 입력과 쌍을 이루는 작업을 설명하는 지침입니다. "
                        "요청을 적절하게 완료하는 응답을 작성합니다.\n\n"
                        "### 지침:\n{instruction}\n\n### 입력:\n{input}\n\n### 응답:"
                    ),
                },
                            "de":{
                    "prompt_input": (
                        "Nachfolgend finden Sie eine Anweisung, die eine Aufgabe beschreibt, gepaart mit einer Eingabe, die weiteren Kontext bereitstellt. "
                        "Schreiben Sie eine Antwort, die die Anfrage angemessen vervollständigt.\n\n"
                        "### Anweisung:\n{instruction}\n\n### Eingang:\n{input}\n\n### Antwort:"
                    ),
                },
                            "cs":{
                    "prompt_input": (
                        "Níže je uvedena instrukce, která popisuje úlohu, spárovaná se vstupem, který poskytuje další kontext. "
                        "Napište odpověď, která vhodně dokončí požadavek.\n\n"
                        "### Návod:\n{instruction}\n\n### Vstup:\n{input}\n\n### Odezva:"
                    )
                },
                            "ru":{
                    "prompt_input": (
                        "Ниже приведена инструкция, описывающая задачу в сочетании с входными данными, предоставляющими дополнительный контекст. "
                        "Напишите ответ, который соответствующим образом дополняет запрос.\n\n"
                        "### Инструкция:\n{instruction}\n\n### Вход:\n{input}\n\n### Ответ:"
                    ),
                },
                            "uk":{
                    "prompt_input": (
                        "Нижче наведено інструкцію, яка описує завдання в поєднанні з введенням, що надає додатковий контекст."
                        "Напишіть відповідь, яка відповідним чином доповнює запит.\n\n"
                        "### Інструкція:\n{instruction}\n\n### Введення:\n{input}\n\n### Відповідь:"
                    )
                },
                            "fr":{
                    "prompt_input": (
                        "Vous trouverez ci-dessous une instruction décrivant une tâche, associée à une entrée fournissant un contexte supplémentaire. "
                        "Écrivez une réponse qui complète de manière appropriée la demande.\n\n"
                        "### Instruction:\n{instruction}\n\n### Saisir:\n{input}\n\n### Réponse:"
                    ),
                },
                            "ja":{
                    "prompt_input": (
                        "B以下は、タスクを説明する指示と、さらなるコンテキストを提供する入力との組み合わせです。 "
                        "リクエストを適切に完了するレスポンスを作成します。\n\n"
                        "### 命令:\n{instruction}\n\n### 入力:\n{input}\n\n### 応答:"
                    )
                },  
                            "ro":{
                    "prompt_input": (
                        "Mai jos este o instrucțiune care descrie o sarcină, asociată cu o intrare care oferă context suplimentar. "
                        "Scrieți un răspuns care completează în mod corespunzător solicitarea.\n\n"
                        "### Instruire:\n{instruction}\n\n### Intrare:\n{input}\n\n### Raspuns:"
                    ),
                    "prompt_no_input": (
                        "Mai jos este o instrucțiune care descrie o sarcină. "
                        "Scrieți un răspuns care completează în mod corespunzător solicitarea.\n\n"
                        "### Instruire:\n{instruction}\n\n### Raspuns:"
                    ),
                },
                            "it":{
                    "prompt_input": (
                        "Di seguito è riportata un'istruzione che descrive un'attività, abbinata a un input che fornisce ulteriore contesto. "
                        "Scrivi una risposta che completa adeguatamente la richiesta.\n\n"
                        "### Istruzioni:\n{instruction}\n\n### Ingresso:\n{input}\n\n### Risposta:"
                    ),
                    "prompt_no_input": (
                        "Di seguito è riportata un'istruzione che descrive un'attività. "
                        "Scrivi una risposta che completa adeguatamente la richiesta.\n\n"
                        "### Istruzioni:\n{instruction}\n\n### Risposta:"
                    ),
                },
                            "nl":{
                    "prompt_input": (
                        "Hieronder vindt u een instructie die een taak beschrijft, gecombineerd met invoer die verdere context biedt. "
                        "Schrijf een antwoord dat het verzoek op passende wijze voltooit.\n\n"
                        "### Instructie:\n{instruction}\n\n### Invoer:\n{input}\n\n### Antwoord:"
                    ),
                    "prompt_no_input": (
                        "Hieronder vindt u een instructie die een taak beschrijft. "
                        "Schrijf een antwoord dat het verzoek op passende wijze voltooit.\n\n"
                        "### Instructie:\n{instruction}\n\n### Antwoord:"
                    ),
                },
}
Multilingul_response = {'en': '### Response:', 
                        'zh': '### 回复:', 
                        'ko': '### 응답:',
                        'de': '### Antwort:',
                        'cs': '### Odezva:',
                        'ru': '### Ответ:',
                        'uk': '### Відповідь:',
                        'fr': '### Réponse:',
                        'ja': '### 応答:',
                        }

Multilingul_instruction = {
    "en": "Translate the following sentences from [SRC] to [TGT].",
    'zh': '将以下句子从 [SRC] 翻译为 [TGT]。', 
    'ko': '다음 문장을 [SRC]에서 [TGT]로 번역하세요.',
    'de': 'Übersetzen Sie die folgenden Sätze von [SRC] in [TGT].',
    'cs': 'Přeložte následující věty z [SRC] do [TGT].',
    'ru': 'Переведите следующие предложения с [SRC] на [TGT].',
    'uk': 'Перекладіть наступні речення з [SRC] на [TGT].',
    'fr': 'Traduisez les phrases suivantes de [SRC] en [TGT].',
    'ja': '次の文を [SRC] から [TGT] に翻訳してください。',
    'ro': 'Traduceți următoarele propoziții din [SRC] în [TGT].',
    'it': 'Traduci le seguenti frasi da [SRC] a [TGT].',
    'nl': 'Vertaal de volgende zinnen van [SRC] naar [TGT].',
}

def get_multilingual_instruct(en_instruct, target_lang):
    return Multilingul_instruction[en_instruct][target_lang]

# Read task instruction, fill in languages
def read_instruct(src, tgt, lang_ins="en"):
    source, target = lang_instruction[lang_ins][src], lang_instruction[lang_ins][tgt]
    ins = Multilingul_instruction[lang_ins].replace("[SRC]", source).replace("[TGT]", target)
    return ins


# Read input data for inference
def read_input(path):
    with open(path, 'r', encoding='utf-8') as f:
        input_data = f.readlines()
    return input_data


# Assembly instruction and input data, handle hints
def create_prompt(instruct, input_data, template="prompt_no_input", prompt_lang="en", few_shot_pre = "", post_ins=False):
    if not post_ins:
        PROMPT = multilingual_PROMPT_DICT[prompt_lang]
    else:
        PROMPT = POST_INS_PROMPT_DICT
    
    if template == "prompt_input":
        list_data_dict = [{"instruction": instruct, "input": p.strip() } for p in input_data]
        prompt_input = PROMPT[template]
        sources = [ prompt_input.format_map(example) for example in list_data_dict ]
    else:
        list_data_dict = [{"instruction": "\n\n".join([instruct, p.strip() ]).strip(), "input": ""} for p in input_data]
        prompt_input = PROMPT[template]
        sources = [ prompt_input.format_map(example) for example in list_data_dict ]
    
    if not few_shot_pre == "":
        sources = [few_shot_pre+item for item in sources ]
    return sources


# Post-process the output, extract translations
def post_process(text, prompt_lang='en'):
    text = text.split(Multilingul_response[prompt_lang])[-1].strip()
    text = text.replace("\n", " ")
    # Cut for contrastive instruction
    if "</p>" in text:
        text = text.split("</p>")[0].split("<p>")[-1]
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name-or-path', type=str, default="PATH/models/ours_models/llama-7b.mt12000", required=False, help='model name in the hub or local path')
    parser.add_argument('--input-file','-i', type=str, default="PATH/data/datasets/iwslt4/test/tst2017en-it.en", required=False, help='input file')
    parser.add_argument('--output-file','-o', type=str, default="PATH/models/ours_models/llama-7b.mt12000/output/en-it.it", required=False, help='output file')
    parser.add_argument('--lang-pair', '-lp', type=str, default='en-it', help='language pair: zh-en, en-de')
    parser.add_argument('--search-algorithm', '-sa', type=str, default='beam', help='search algorithms: sample, beam')
    parser.add_argument('--batch', '-b', type=int, default=2, help='batch size')
    parser.add_argument('--template', '-tp', type=int, default=1, help='0: prompt_no_input, 1: prompt_input')
    parser.add_argument('--temperature', '-t', type=float, default=0.1, help='temperature: 0.7 for text generation')
    parser.add_argument('--prompt-lang', '-pl', type=str, default="en", help='prompt language for generation')
    parser.add_argument('--n-shot', '-n', type=int, default=0, help='inference with n demonstrative samples')
    parser.add_argument('--post-ins', type=bool, default=False, help='inference with post-ins prompt')
    args = parser.parse_args()
    model_name_or_path = args.model_name_or_path
    input_file = args.input_file
    output_file = args.output_file
    lang_pair = args.lang_pair
    search = args.search_algorithm
    batch = args.batch
    temperature = args.temperature
    temp = args.template
    post_ins = args.post_ins
    template = "prompt_input" if temp > 0 else "prompt_no_input"
    prompt_lang = args.prompt_lang
    n_shot = args.n_shot
    # Load checkpoints
    model = LLM(model_name_or_path, tensor_parallel_size=1, gpu_memory_utilization=0.9, swap_space=64, enforce_eager=True)

    to_use_fast = False
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=to_use_fast)

    sampling_param = SamplingParams(
        top_p=1, temperature=0, max_tokens=512, 
        use_beam_search=True, best_of=4, stop_token_ids=[tokenizer.eos_token_id], 
        )
    # Prepare input data
    srcl, tgtl = lang_pair.split('-')
    instruct = read_instruct(srcl, tgtl, lang_ins = prompt_lang)
    input_data = read_input(input_file)
    if n_shot == 0:
        few_shot_pre = ""
    else:
        file = f"./{str(n_shot)}_shot_samples.json"
        all_few_shot_pre = json.load(open(file))
        few_shot_pre = all_few_shot_pre[lang_pair]
    prompt = create_prompt(instruct, input_data, template, prompt_lang, few_shot_pre, post_ins)
    print(f"example: {prompt[0]}")

    # Generate
    torch.manual_seed(0)
    with open(output_file, 'w', encoding='utf-8') as fo,open(output_file+".hyp", 'w', encoding='utf-8') as fo2:
        decoded_tokens = model.generate(prompt, sampling_params=sampling_param)
        for item in decoded_tokens:
            print(item.outputs[0].text.replace("\n", ""), file=fo2, flush=True)
        for sample in prompt:
            print(sample, file=fo, flush=True)
