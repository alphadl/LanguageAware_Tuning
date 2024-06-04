import re
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import inspect
import torch
import transformers
import utils
from torch import nn
from torch.utils.data import Dataset
from transformers import Trainer

import os
os.environ["WANDB_DISABLED"] = "true"

import random
random.seed(1234)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

PROMPT_DICT = {
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
}
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


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    negative_instruction: bool = field(
        default=False,
        metadata={"help": "whether using the random selected negative instruction"},
    )
    post_instruction: bool = field(
        default=False,
        metadata={"help": "whether using the post instruction prompt"},
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    ours: bool = field(
        default=False,
        metadata={"help": "whether using unlikelihood loss to train model"},
    )
    alpha: float = field(
        default=0.05,
        metadata={"help": "alpha for unlikelihood loss"},
    )

    
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]

    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, post_ins: bool):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        if post_ins:
            print(">> post instruction prompt for training")
            prompt_input, prompt_no_input = POST_INS_PROMPT_DICT["prompt_input"], POST_INS_PROMPT_DICT["prompt_no_input"]
        else:
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, post_ins=data_args.post_instruction)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


class CustomSupervisedDataset(Dataset):
    """Dataset for unlikelihood tuning."""
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, negative_instruction = False):
        super(CustomSupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        logging.warning(">> alpaca_template for training")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        negative_sources = []
        if negative_instruction:
            logging.warning(">> negative instruction ")

            for example in list_data_dict:
                while True:
                    negative_ins = random.choice(list_data_dict)["instruction"]
                    if not example["instruction"] == negative_ins:
                        break
                example["instruction"] = negative_ins
                example = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
                negative_sources.append(example)

            # negative data dict
            logging.warning("Tokenizing negative inputs... This may take some time...")
            negative_data_dict = preprocess(negative_sources, targets, tokenizer)
            self.negative_input_ids = negative_data_dict["input_ids"]
            self.negative_labels = negative_data_dict["labels"]
        else: 
            raise NotImplementedError("This function has not been implemented.")

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], 
                    labels=self.labels[i], 
                    negative_input_ids= self.negative_input_ids[i],
                    negative_labels = self.negative_labels[i],
                    )


@dataclass
class CustomDataCollatorForSupervisedDataset(object):
    """Collate for unlikelihood tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        negative_input_ids, negative_labels = tuple([instance[key] for instance in instances] for key in ("negative_input_ids", "negative_labels"))
        negative_input_ids = torch.nn.utils.rnn.pad_sequence(
            negative_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        negative_labels = torch.nn.utils.rnn.pad_sequence(negative_labels, batch_first=True, padding_value=IGNORE_INDEX)


        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            negative_input_ids=negative_input_ids,
            negative_labels=negative_labels,
            negative_att_mask=negative_input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_custom_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for unlikelihood tuning."""
    train_dataset = CustomSupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, negative_instruction=data_args.negative_instruction)
    data_collator = CustomDataCollatorForSupervisedDataset(tokenizer=tokenizer)
    print(train_dataset[0], train_dataset[1], train_dataset[2])
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

class CustomTrainer(Trainer):
    """
    trainer for unlikelihood tuning 
    """
    def compute_loss(self, model, inputs, return_outputs=False): 
        labels = inputs.get("labels") 
        negative_labels = inputs.get("negative_labels") 
        # forward pass 
        outputs = model( 
            input_ids = inputs["input_ids"], 
            attention_mask = inputs["attention_mask"] 
            ) 
        # working with batch size per device equal to 1
        negative_outputs = model(
            input_ids = inputs["negative_input_ids"], 
            attention_mask = inputs["negative_att_mask"]
            )
        logits = outputs.get("logits")

        # compute likelihood loss of cross entropy loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()

        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        loss = loss_fct(shift_logits, shift_labels)
        
        negative_loss = torch.tensor(0)
        if not inputs["negative_labels"].size()[1] == 1:
            negative_loss = self.compute_negative_loss(negative_outputs, negative_labels)

        # print(f">> likelihood loss: {loss}, unlikelihood loss: {negative_loss}" )
        loss += self.args.alpha * negative_loss
        return (loss, outputs) if return_outputs else loss

    def compute_negative_loss(self, negative_outputs, negative_labels):

        logits = negative_outputs.get("logits")
        # compute unlikelihood loss of cross entropy loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = negative_labels[..., 1:].contiguous()

        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        m = nn.Softmax(dim=-1)
        loss_fn = nn.NLLLoss()

        probs = m(shift_logits)
        one_minus_probs = torch.log(torch.clamp((1.0 - probs), min=1e-9))
        negative_loss = loss_fn(
            one_minus_probs,
            shift_labels
        )
        if negative_loss == float("inf") or negative_loss ==  float("nan"):
            print("unusual unlikelihood loss: ", negative_loss)
            return torch.tensor(0)
        return negative_loss
    
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))

            # do not remove unlikelihood loss related item 
            self._signature_columns.append("negative_input_ids")
            self._signature_columns.append("negative_labels")


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args, data_args, training_args)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    if not training_args.ours:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    else:
        data_module = make_custom_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        trainer = CustomTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
