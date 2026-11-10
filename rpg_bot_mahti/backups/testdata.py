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


class DialogueDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def tokenize(text, max_length):
    dialogue_turns = text.split("\n")
    input_ids = []
    for turn in dialogue_turns:
        turn_ids = tokenizer.encode(turn, add_special_tokens=True)
        input_ids.extend(turn_ids)
        input_ids.append(tokenizer.eos_token_id)
    input_ids = input_ids[:-1]  # remove the last EOS token
    pad_length = max_length - len(input_ids)
    input_ids = input_ids + [tokenizer.pad_token_id] * pad_length
    return torch.tensor(input_ids)


def collate(
    examples: List[Tuple[str, str]],
):
    max_prompt_length = 50
    max_completion_length = 50
    prompts = [e[0] for e in examples]
    completions = [e[1] for e in examples]

    tokenized_prompts = [tokenize(p, max_prompt_length) for p in prompts]
    tokenized_completions = [tokenize(c, max_completion_length) for c in completions]

    inputs = []
    labels = []

    for p, c in zip(tokenized_prompts, tokenized_completions):
        p = p[:max_prompt_length]
        c = c[:max_completion_length]

        if len(p) < max_prompt_length:
            p = pad_sequence(p, max_prompt_length)
        if len(c) < max_completion_length:
            c = pad_sequence(c, max_completion_length)

        inputs.append(p)
        labels.append(c)

    return torch.stack(inputs), torch.stack(labels)


import json
from torch.utils.data import DataLoader, IterableDataset


train_dataset = "data/NLG_RPG.jsonl"
samples = []
with open(train_dataset) as f:
    for sample_line in f:
        sample = json.loads(sample_line)
        prompt = sample["prompt"]
        completion = sample["completion"]
        samples.append((prompt, completion))
dataset = DialogueDataset(samples)

train_sampler = torch.utils.data.SequentialSampler(dataset)
train_dataloader = DataLoader(
    dataset, batch_size=1, sampler=train_sampler, collate_fn=collate
)
for elem in train_dataloader:
    print(elem)
