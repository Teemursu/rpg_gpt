import logging
import os

import torch
import json
from torch.utils.data import DataLoader, IterableDataset

from transformers import (
    GPT2Tokenizer,
    GPTJForCausalLM,
    Trainer,
    TrainingArguments,
)
import pandas as pd

logger = logging.getLogger(__name__)
import argparse
import deepspeed
from accelerate import Accelerator
from torch.utils.data.distributed import DistributedSampler


from torch.utils.data import DataLoader, Dataset

import pickle
from typing import List

accelerator = Accelerator()


parser = argparse.ArgumentParser()
# handle any own command line arguments here
parser = deepspeed.add_config_arguments(parser)
parser.add_argument("--local_rank", type=int, default=0)
args = parser.parse_args()

os.environ["TRANSFORMERS_CACHE"] = "/cache"

# local_rank = int(os.environ["LOCAL_RANK"])
# torch.cuda.set_device(local_rank)
deepspeed.init_distributed()
# torch.distributed.init_process_group(backend="nccl")
torch.distributed.barrier()
torch.cuda.set_device(torch.cuda.current_device())
ddp_params = {"num_losses": 1}

device_ids = [
    torch.device("cuda:0"),
    torch.device("cuda:1"),
    torch.device("cuda:2"),
    torch.device("cuda:3"),
]

model_type = "gpt-j-6B"

# Load tokenizer and model
max_length = 50
truncate = True
tokenizer = GPT2Tokenizer.from_pretrained(
    "EleutherAI/gpt-j-6B",
    cache_dir="cached",
    mlm=False,
    max_length=max_length
    # truncation=True,
    # padding="max_length",
)
model = GPTJForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B",
    cache_dir="cached",
    revision="float16",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

# Add special tokens to tokenizer
num_added_toks = tokenizer.add_special_tokens(
    {
        "pad_token": "[PAD]",
        "additional_special_tokens": [
            "<|persuade|>",
            "<|computer|>",
            "<|forcepersuade|>",
            "<|insight|>",
            "<|might|>",
            "<|perception|>",
            "<|awareness|>",
            "<|repair|>",
            "<|resolve|>",
            "<|intimidate|>",
            "<|magic|>",
            "<|constitution|>",
            "<|dexterity|>",
            "<|wisdom|>",
            "<|demolitions|>",
            "<|treatinjury|>",
            "<|bows|>",
            "<|bluff|>",
            "<|survival|>",
            "<|security|>",
            "<|swords|>",
            "<|mechanics|>",
            "<|perform|>",
            "<|playername|>",
        ],
    }
)


def collate(examples: List[torch.Tensor]):
    if truncate:
        examples = [ex[:max_length] for ex in examples]
    examples = [
        torch.cat(
            (
                ex,
                torch.full(
                    (max_length - len(ex),), tokenizer.pad_token_id, dtype=torch.long
                ),
            ),
            0,
        )
        if len(ex) < max_length
        else ex
        for ex in examples
    ]
    return torch.stack(examples)


train_dataset = "data/NLG_RPG.jsonl"
samples = []
with open(train_dataset) as f:
    for sample_line in f:
        sample = json.loads(sample_line)
        samples.append(sample)
train_dataset = samples
encoded_data = []
for example in train_dataset:
    prompt = example["prompt"]
    completion = example["completion"]
    input_text = prompt + completion
    encoded_input = tokenizer.encode(input_text, return_tensors="pt")
    encoded_data.append(encoded_input)
# )

encoded_data = []
for example in train_dataset:
    prompt = example["prompt"]
    completion = example["completion"]
    input_text = prompt + completion
    encoded_input = tokenizer.encode(input_text, return_tensors="pt")
    encoded_data.append(encoded_input)

# train_dataloader, model = accelerator.prepare(
#    train_dataloader, model
# )
# Now proceed as normal, plus pass the deepspeed config file
training_args = TrainingArguments(
    num_train_epochs=3,
    output_dir="results",
    deepspeed="/scratch/project_2001403/poyhnent/configs/ds_config.json",
    per_device_train_batch_size=1,
)

trainer = Trainer(
    data_collator=collate,
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    create_optimizers=(None, None),
)

trainer.train()
trainer.save_model()
