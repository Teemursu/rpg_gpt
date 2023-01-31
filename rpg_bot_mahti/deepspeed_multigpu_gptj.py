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
import deepspeed
from accelerate import Accelerator
from torch.utils.data.distributed import DistributedSampler

from torch.nn.utils.rnn import pad_sequence


from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

import pickle
from typing import Dict, List, Tuple

from tqdm import tqdm

accelerator = Accelerator()

from torch.utils.data import DataLoader

torch.cuda.manual_seed_all(42)


parser = argparse.ArgumentParser()
# handle any own command line arguments here
parser = deepspeed.add_config_arguments(parser)
parser.add_argument("--local_rank", type=int, default=0)
args = parser.parse_args()

os.environ["TRANSFORMERS_CACHE"] = "/cache"

# local_rank = int(os.environ["LOCAL_RANK"])
# torch.cuda.set_device(local_rank)
deepspeed.init_distributed()
# deepspeed.init_process_group("nccl")
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
model = GPTJForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B",
    cache_dir="cached",
    revision="float16",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
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
optimizer = torch.optim.SGD(model.parameters(), lr=training_args.learning_rate)
torch.cuda.empty_cache()


class DialogueDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def tokenize(text, max_length):
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    pad_length = max_length - len(input_ids)
    input_ids = input_ids + [tokenizer.pad_token_id] * pad_length
    return torch.tensor(input_ids)


def collate(examples: List[torch.Tensor]):
    prompts = [e[0] for e in examples]
    completions = [e[1] for e in examples]

    max_length = 50

    tokenized_prompts = [tokenize(p, max_length) for p in prompts]
    tokenized_completions = [tokenize(c, max_length) for c in completions]

    truncated_prompts = []
    truncated_completions = []

    for p, c in zip(tokenized_prompts, tokenized_completions):
        combined_text = p + c
        if len(combined_text) > max_length:
            diff = (
                len(combined_text) - max_length + 1
            )  # +1 to not truncate the pad token
            p = p[:-diff]
            combined_text = p + c

        truncated_prompts.append(p)
        truncated_completions.append(c)

    return torch.stack(truncated_prompts), torch.stack(truncated_completions)


import json
from torch.utils.data import DataLoader, IterableDataset


train_dataset = "data/NLG_RPG.jsonl"
samples = []
with open(train_dataset) as f:
    for sample_line in f:
        sample = json.loads(sample_line)
        samples.append(sample)
# train_dataset = samples
dataset = DialogueDataset(samples)
# )

encoded_data = []
for example in train_dataset:
    prompt = example["prompt"]
    completion = example["completion"]
    input_text = prompt + completion
    encoded_input = tokenizer.encode(input_text, return_tensors="pt")
    encoded_data.append(encoded_input)

train_sampler = torch.utils.data.SequentialSampler(encoded_data)
train_dataloader = DataLoader(
    encoded_data, batch_size=1, sampler=train_sampler, collate_fn=collate
)


torch.cuda.empty_cache()
num_epochs = 3
num_training_steps = num_epochs * len(train_dataset)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
torch.cuda.empty_cache()
progress_bar = tqdm(range(num_training_steps))

train_dataloader, model, optimizer, lr_scheduler = accelerator.prepare(
    train_dataloader, model, optimizer, lr_scheduler
)
torch.cuda.empty_cache()
gradient_accumulation_steps = 1
model.train()
for epoch in range(num_epochs):
    torch.cuda.empty_cache()
    accumulated_loss = 0
    num_accumulated_steps = 0
    for batch in train_dataloader:
        prompts, completions = batch
        outputs = model(prompts, labels=completions)
        loss = outputs.loss
        accumulated_loss += loss
        num_accumulated_steps += 1
        torch.cuda.empty_cache()

        if num_accumulated_steps == gradient_accumulation_steps:
            accumulated_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            accumulated_loss = 0
            num_accumulated_steps = 0
            torch.cuda.empty_cache()

    progress_bar.update(1)

    # handle any leftover steps
    if num_accumulated_steps > 0:
        accumulated_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()

# torch.save(
#    {
#        "model_state_dict": model.state_dict(),
#        "optimizer_state_dict": optimizer.state_dict(),
#        "scheduler_state_dict": lr_scheduler.state_dict(),
#    },
#    "/scratch/project_2001403/poyhnent/results/gpt_rpg_gptj_6b.pt",
# )
# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = (
    model.module if hasattr(model, "module") else model
)  # Take care of distributed/parallel training
model_to_save.save_pretrained("gptj_rpg")
tokenizer.save_pretrained("gptj_rpg")
