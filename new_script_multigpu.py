import glob
import logging
import os
import pickle
import random
import re
import shutil
import torch
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import transformers
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
logger = logging.getLogger(__name__)

import argparse
import deepspeed


parser = argparse.ArgumentParser()
# handle any own command line arguments here
parser = deepspeed.add_config_arguments(parser)
parser.add_argument("--local_rank", type=int, default=0)
args = parser.parse_args()

os.environ['TRANSFORMERS_CACHE'] = '/cache'

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
deepspeed.init_distributed()
#torch.distributed.init_process_group(backend="nccl")
torch.distributed.barrier() 
torch.cuda.set_device(torch.cuda.current_device())
ddp_params = {"num_losses": 1}

device_ids = [
    torch.device("cuda:0"),
    torch.device("cuda:1"),
    torch.device("cuda:2"),
    torch.device("cuda:3"),
]

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B",
                                          cache_dir="cached")
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B", 
                                          #torch_dtype=torch.float16,
                                          cache_dir="cached")

# Add special tokens to tokenizer
num_added_toks = tokenizer.add_special_tokens(
        {
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
            ]
        }
    )
#print(f"Added {num_added_toks} special tokens: {special_tokens}")

# Prepare data
# Assuming df has columns "response", "context/0", "context/1", ..., "context/8"
df = "data/nlg_data_10_turns_all_data_without_consoledialogue.csv"
df = pd.read_csv(df, sep="\t")
df.drop("Unnamed: 0.1", axis=1, inplace=True)
df.drop("Unnamed: 0", axis=1, inplace=True)
df = df.drop_duplicates()
df.drop("game", axis=1, inplace=True)
df = df[df.columns[: list(df.columns).index("context/3") + 1]]
examples = []

def construct_conv(row, tokenizer, eos=True):
    flatten = lambda l: [item for sublist in l for item in sublist]
    conv = list(
        reversed([tokenizer.encode(str(x)) + [tokenizer.eos_token_id] for x in row])
    )
    conv = flatten(conv)
    return conv

for _, row in df.iterrows():
        conv = construct_conv(row, tokenizer)
        examples.append(conv)
        
print(examples)
# Prepare training arguments
training_args = TrainingArguments(
    output_dir='results',          # output directory
    evaluation_strategy = "steps", 
    cache_dir = "cached",
    eval_steps = 50,
    per_device_train_batch_size=1,  # batch size
    per_device_eval_batch_size=1, 
    num_train_epochs=1,             # number of training epochs
    weight_decay=0.01,              # weight decay
    learning_rate=3e-5,             # learning rate
    warmup_steps=500,               # warmup steps
    logging_dir='logs',           # directory for storing logs
    logging_steps=50,
    save_steps=50,
    
    deepspeed="configs/ds_config.json"
)

# Prepare optimizer
optimizer = torch.optim.Lamb(model.parameters(), lr=training_args.learning_rate)

# Create trainer
trainer = Trainer(
    model=model,                        
    args=training_args,                 
    train_dataset=examples,
    #eval_dataset=val_df,
    data_collator = transformers.data.data_collator.DataCollatorForLanguageModeling(tokenizer=tokenizer),
    optimizer=optimizer
)

# Add parallel and distributed training
if local_rank != -1:
    trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=examples,
    #eval_dataset=val_df,
    data_collator = transformers.data.data_collator.DataCollatorForLanguageModeling(tokenizer=tokenizer),
    optimizer=optimizer
    )
    transformers.DistributedTrainer(trainer, device='cuda', local_rank=local_rank)

trainer.train()
trainer.save_model('./results')