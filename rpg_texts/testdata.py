import logging
import os

import torch

import transformers
from transformers import (
    get_scheduler,
    GPT2Tokenizer,
    GPTJForCausalLM,
    Trainer,
    TrainingArguments,
)
import pandas as pd

logger = logging.getLogger(__name__)
import argparse
from accelerate import Accelerator
from torch.utils.data.distributed import DistributedSampler

from torch.nn.utils.rnn import pad_sequence


from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

import pickle
from typing import Dict, List, Tuple

from tqdm import tqdm


from torch.utils.data import DataLoader

torch.cuda.manual_seed_all(42)


parser = argparse.ArgumentParser()
# handle any own command line arguments here
parser.add_argument("--local_rank", type=int, default=0)
args = parser.parse_args()

os.environ["TRANSFORMERS_CACHE"] = "/cache"

# local_rank = int(os.environ["LOCAL_RANK"])
# torch.cuda.set_device(local_rank)
# torch.distributed.init_process_group(backend="nccl")
# torch.distributed.barrier()
# torch.cuda.set_device(torch.cuda.current_device())
# ddp_params = {"num_losses": 1}

device_ids = [
    torch.device("cuda:0"),
    torch.device("cuda:1"),
    torch.device("cuda:2"),
    torch.device("cuda:3"),
]

model_type = "gpt-j-6B"

# Load tokenizer and model
max_length = 100
truncate = True
tokenizer = GPT2Tokenizer.from_pretrained(
    "EleutherAI/gpt-j-6B",
    cache_dir="cached",
    mlm=False,
    max_length=max_length
    # truncation=True,
    # padding="max_length",
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

torch.cuda.empty_cache()
# print(examples)
# Prepare training arguments
training_args = TrainingArguments(
    output_dir="/scratch/project_2001403/poyhnent/results",  # output directory
    evaluation_strategy="steps",
    # cache_dir = "cached",
    eval_steps=50,
    per_device_train_batch_size=1,  # batch size
    per_device_eval_batch_size=1,
    num_train_epochs=1,  # number of training epochs
    weight_decay=0.01,  # weight decay
    learning_rate=3e-5,  # learning rate
    warmup_steps=500,  # warmup steps
    logging_dir="logs",  # directory for storing logs
    logging_steps=50,
    save_steps=50,
    deepspeed="/scratch/project_2001403/poyhnent/configs/ds_config.json",
)


# train_batch_size = 1 * max(1, args.n_gpu)
# Prepare optimizer
torch.cuda.empty_cache()


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


import json
from torch.utils.data import DataLoader, IterableDataset


class JsonDataset(IterableDataset):
    def __init__(self, files):
        self.files = files

    def __iter__(self):
        for json_file in self.files:
            with open(json_file) as f:
                for sample_line in f:
                    sample = json.loads(sample_line)
                    yield sample["x"], sample["time"], ...


train_dataset = JsonDataset(
    [
        {
            "prompt": "RPGBOT: The old roads tell of such things - to any who care to walk them.\nRPGBOT: Then you have not yet met the road, and it will humble you long before you reach its end.\n",
            "completion": "If you ever broke your oath like that for my sake, I would rather walk on alone than see you give up your word for me. That is not who you are, or who you could be.",
        },
        {
            "prompt": "RPGBOT: The border wars proved such things. In only one knight did I see courage.\nRPGBOT: Then you have not yet met the road, and it will humble you long before you reach its end.\n",
            "completion": "If you ever broke your oath like that for my sake, I would rather walk on alone than see you give up your word for me. That is not who you are, or who you could be.",
        },
    ]
)

# train_sampler = torch.utils.data.SequentialSampler(train_dataset)
train_dataloader = DataLoader(
    train_dataset, batch_size=1, collate_fn=collate
)  # sampler=train_sampler)

train_sampler = torch.utils.data.SequentialSampler(train_dataset)
train_dataloader = DataLoader(
    train_dataset, collate_fn=collate, batch_size=1, sampler=train_sampler
)
for i, example in enumerate(train_dataloader):
    example_len = example.shape[1]
    if example_len > max_length:
        max_length = example_len

print(
    "Longest instance is length",
    max_length,
)
