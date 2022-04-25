#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on multiple choice relying on the accelerate library without using a Trainer.
"""
# This script is extended from huggingface Transformers

import argparse
import logging
from dataclasses import dataclass
from itertools import chain
import os
import random
from typing import Optional, Union
import json
from pathlib import Path



import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)

from transformers.utils import PaddingStrategy, get_full_repo_name

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def parse_args():
    parser = argparse.ArgumentParser(description="Predict a transformers model on a text classification task")
    parser.add_argument(
        "--test_file", type=str, default="./cache/test.csv", help="A csv or a json file containing the testing data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="./ckpt/multi_choice/",
        help="Path to pretrained model or model identifier from huggingface.co/models for multiple choice.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default="./cache/multi_choice.json",
        help="Output file of prediction of the muliple choice."
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    args = parser.parse_args()  
    
    return args


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        # for feature in features:
        #     print(feature)
        # exit()
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features 
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}

        return batch

def postprocessAnswer(predictions, dataset, args):

    ending_names = [f"ending{i}" for i in range(4)]

    output_json = []
    for data, predict in zip(dataset, predictions):
        one_output = {}
        one_output["id"] = data["fold-ind"]
        one_output["title"] = "custom"
        one_output["context"] = data[ending_names[predict]]
        one_output["question"] = data["sent1"]

        output_json.append(one_output)

    output_json = {"data": output_json}
    args.output_file.write_text(json.dumps(output_json, indent = 2, ensure_ascii=False))

    logging.info(f"multiple choice prediction saved at {str(args.output_file.resolve())}")




def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)


    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)


    accelerator.wait_for_everyone()

    data_files = {}
    data_files["test"] = args.test_file
    extension = args.test_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)
    column_names = raw_datasets["test"].column_names
    

    ending_names = [f"ending{i}" for i in range(4)]
    context_name = "sent1"
    question_header_name = "sent2"

    config = AutoConfig.from_pretrained(args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)

    model = AutoModelForMultipleChoice.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )

    model.resize_token_embeddings(len(tokenizer))

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        first_sentences = [[context] * 4 for context in examples[context_name]]
        question_headers = examples[question_header_name]
        second_sentences = [
            [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
        ]

        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            max_length=args.max_length,
            padding=padding,
            truncation=True,
        )
        # Un-flatten
        tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        tokenized_inputs["id"] = examples["fold-ind"]
        return tokenized_inputs

    with accelerator.main_process_first():

        processed_datasets = raw_datasets.map(
            preprocess_function, batched=True, remove_columns=raw_datasets["test"].column_names
        )


    test_dataset = processed_datasets["test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(test_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {test_dataset[index]}.")

    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForMultipleChoice(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    dataset_for_model = test_dataset.remove_columns(["id"])
    test_dataloader = DataLoader(dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    
    device = accelerator.device
    model.to(device)

    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )


    logger.info("***** Running testing *****")
    logger.info(f"  Num examples = {len(test_dataset)}")

    model.eval()
    predictions = []
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        with torch.no_grad():
            outputs = model(**batch)
        predictions.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
        
    postprocessAnswer(predictions, raw_datasets["test"], args)



    
if __name__ == "__main__":
    main()

