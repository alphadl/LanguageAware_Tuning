import logging
from typing import Set, List, Union, Tuple, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, LogitsProcessorList

from scripts.utils_run import FLORES101_CONVERT
from translation_models import TranslationModel
from translation_models.m2m100 import EnsembleLogitsProcessor
from translation_models.utils_llama import language_names, one_shot_sentences


class LLaMaMTTranslationModel(TranslationModel):

    # # Official templates used during instruction tuning of LLaMA
    # TEMPLATE_0 = "{src_sent}\n\nTranslate to {tgt_lang}"
    # TEMPLATE_1 = "{src_sent}\n\nCould you please translate this to {tgt_lang}?"
    # TEMPLATE_2 = "{src_sent}\n\nTranslate this to {tgt_lang}?"
    # TEMPLATE_3 = "Translate to {tgt_lang}:\n\n{src_sent}"
    # TEMPLATE_4 = "Translate the following sentence to {tgt_lang}:\n{src_sent}"
    # TEMPLATE_5 = "How is \"{src_sent}\" said in {tgt_lang}?"
    # TEMPLATE_6 = "Translate \"{src_sent}\" to {tgt_lang}?"

    # 1. use Alpaca instruction following format Prompt
    SYSTEM_PROMPT =  (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nTranslate the following sentences from {src_lang} to {tgt_lang}.\n\n### Input:\n{src_sent}\n\n### Response:"
    )


    def __init__(self,
                 model_name_or_path: str,
                 message_template: str = None,
                 one_shot: bool = False,
                 **kwargs,
                 ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto', load_in_4bit=True,
                                                          torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.pipeline = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)
        self.message_template = message_template
        self.one_shot = one_shot
        self.src_lang = None
        self.tgt_lang = None

    def __str__(self):
        return str(self.model_name_or_path).replace("/", "_")

    @property
    def supported_languages(self) -> Set[str]:
        return {code for code, code3 in FLORES101_CONVERT.items() if code3 in language_names}

    def requires_src_lang(self):
        return True

    def _set_src_lang(self, src_lang: str):
        assert src_lang in self.supported_languages
        self.src_lang = src_lang

    def _set_tgt_lang(self, tgt_lang: str):
        assert tgt_lang in self.supported_languages
        self.tgt_lang = tgt_lang

    def _lang_code_to_name(self, lang_code: str) -> str:
        lang_code3 = FLORES101_CONVERT.get(lang_code, lang_code)
        return language_names[lang_code3]

    @torch.no_grad()
    def _translate(self,
                   source_sentences: List[str],
                   return_score: bool = False,
                   batch_size: int = 1,
                   num_beams: int = 1,
                   **kwargs,
                   ) -> Union[List[str], List[Tuple[str, float]]]:
        if return_score:
            raise NotImplementedError
        if batch_size != 1:
            logging.warning(
                f"Batch size {batch_size} is not supported by LLaMaTranslationModel. Setting batch size to 1.")
            batch_size = 1
        if num_beams != 1:
            logging.warning(f"Beam search is not supported by LLaMaTranslationModel. Setting num_beams to 1.")
            num_beams = 1

        assert self.src_lang is not None
        assert self.tgt_lang is not None
        system_prompt = self.SYSTEM_PROMPT.format(
            src_lang=self._lang_code_to_name(self.src_lang),
            tgt_lang=self._lang_code_to_name(self.tgt_lang),
        )

        if self.one_shot:
            raise NotImplementedError
            # system_prompt += "\n\nExample instruction:\n{instruction}\n\nExample response:\nSure, here's the translation:\n{response}".format(
            #     instruction=self.message_template.format(
            #         src_lang=self._lang_code_to_name(self.src_lang),
            #         tgt_lang=self._lang_code_to_name(self.tgt_lang),
            #         src_sent=one_shot_sentences[FLORES101_CONVERT.get(self.src_lang, self.src_lang)],
            #     ),
            #     response=one_shot_sentences[FLORES101_CONVERT.get(self.tgt_lang, self.tgt_lang)],
            # )

        translations = []
        for source_sentence in tqdm(source_sentences):
            # prompt_template = PromptTemplate(system_prompt=system_prompt)
            # message = self.message_template.format(
            #     src_lang=self._lang_code_to_name(self.src_lang),
            #     tgt_lang=self._lang_code_to_name(self.tgt_lang),
            #     src_sent=source_sentence,
            # )
            # logging.info(message)
            # prompt_template.add_user_message(message)
            # prompt = prompt_template.build_prompt()
            # prompt += "Sure, here's the translation:\n"

            prompt = self.SYSTEM_PROMPT.format(
                src_lang=self._lang_code_to_name(self.src_lang),
                tgt_lang=self._lang_code_to_name(self.tgt_lang),
                src_sent=source_sentence,
            )
            logging.info(prompt)
            inputs = self.pipeline.preprocess(prompt)
            output = self.pipeline.forward(
                inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                max_length=1200,  # Max ref length across Flores-101 is 960
                remove_invalid_values=True,
                num_beams=num_beams,
                # Disable sampling
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )
            output = self.pipeline.postprocess(output)
            output = output[0]['generated_text']
            logging.info(output)
            _, output = output.rsplit("### Response:", maxsplit=1)
            # prompt_template.add_model_reply(output, includes_history=True)
            # response = prompt_template.get_model_replies(strip=True)[0]
            # response_lines = response.replace("Sure, here's the translation:", "").strip().split("\n")
            # if not response_lines:
            #     translation = ""
            # else:
            #     translation = response_lines[0].strip()
            translation = output.replace("\n", "")
            translations.append(translation)
        return translations

    def _translate_multi_source(self,
                                multi_source_sentences: List[str],
                                src_langs: List[str],
                                tgt_langs: List[str],
                                src_weights: Optional[List[float]] = None,
                                num_beams: int = 1,
                                **kwargs,
                                ) -> str:
        assert len(multi_source_sentences) == len(src_langs) == len(tgt_langs)
        if src_weights is not None:
            assert len(src_weights) == len(multi_source_sentences)
        if num_beams != 1:
            logging.warning(f"Beam search is not supported by LLaMaTranslationModel. Setting num_beams to 1.")
            num_beams = 1

        prompts = []
        for src_sent, src_lang, tgt_lang in zip(multi_source_sentences, src_langs, tgt_langs):
            prompt = self.SYSTEM_PROMPT.format(
                src_lang=self._lang_code_to_name(src_lang),
                tgt_lang=self._lang_code_to_name(tgt_lang),
                src_sent=src_sent,
            )
            prompts.append(prompt)
            
        # logging.info(prompts)
        inputs = [self.pipeline.preprocess(prompt) for prompt in prompts]
        input_ids = [x['input_ids'][0].tolist() for x in inputs]
        attention_mask = [x['attention_mask'][0].tolist() for x in inputs]
        # Add left padding
        max_len = max(len(x) for x in input_ids)
        pad_token_id = self.tokenizer.get_vocab()["▁"] if "▁" in self.tokenizer.get_vocab().keys() else self.tokenizer.get_vocab()["[PAD]"]
        input_ids = [[pad_token_id] * (max_len - len(x)) + x for x in input_ids]
        attention_mask = [[0] * (max_len - len(x)) + x for x in attention_mask]
        input_ids = torch.tensor(input_ids).to(self.model.device)
        attention_mask = torch.tensor(attention_mask).to(self.model.device)
        logits_processor = LogitsProcessorList([
            EnsembleLogitsProcessor(num_beams=num_beams, source_weights=src_weights),
        ])
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=1200,
            logits_processor=logits_processor,
            remove_invalid_values=True,
            # Disable sampling
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            **kwargs,
        )
        output = output.reshape(1, output.shape[0], *output.shape[1:])
        output = {
            "generated_sequence": output,
            "input_ids": input_ids[0],
            "prompt_text": '',
        }
        output = self.pipeline._ensure_tensor_on_device(output, device=torch.device("cpu"))
        output = self.pipeline.postprocess(output)
        output = output[0]['generated_text']
        _, output = output.rsplit("### Response:", maxsplit=1)
        logging.info(output)
        translation = output.replace("\n", "")
        return translation

