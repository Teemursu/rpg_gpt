import logging
import os

import torch

import transformers
from transformers import (
    get_scheduler,
    GPT2Tokenizer,
    GPTNeoForCausalLM,
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

import pickle
from typing import Dict, List, Tuple

from tqdm import tqdm

accelerator = Accelerator()

from torch.utils.data import DataLoader

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

model_type = "gpt_neo_2.7b"

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(
    "EleutherAI/gpt-neo-2.7B",
    cache_dir="cached",
    mlm=False,
    max_length=10,
    truncation=True,
)
model = GPTNeoForCausalLM.from_pretrained(
    "EleutherAI/gpt-neo-2.7B",
    cache_dir="cached",
    torch_dtype=torch.float16,
)

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
# print(f"Added {num_added_toks} special tokens: {special_tokens}")

# Prepare data
# Assuming df has columns "response", "context/0", "context/1", ..., "context/8"
df = "data/nlg_data_10_turns_all_data_without_consoledialogue.csv"
df = pd.read_csv(df, sep="\t")
df.drop("Unnamed: 0.1", axis=1, inplace=True)
df.drop("Unnamed: 0", axis=1, inplace=True)
df = df.drop_duplicates()
df.drop("game", axis=1, inplace=True)
df = df[df.columns[: list(df.columns).index("context/3") + 1]]

# trn_df, val_df = train_test_split(df, test_size=0.1)
trn_df = df
# trn_df.head()


def construct_conv(row, tokenizer, eos=True):
    flatten = lambda l: [item for sublist in l for item in sublist]
    conv = list(
        reversed([tokenizer.encode(str(x)) + [tokenizer.eos_token_id] for x in row])
    )
    conv = flatten(conv)
    print(conv)
    return conv


class ConversationDataset(Dataset):
    def __init__(self, tokenizer: GPT2Tokenizer, df, block_size=512):
        overwrite_cache = True
        block_size = block_size - (
            tokenizer.model_max_length - tokenizer.max_len_single_sentence
        )

        directory = "cached"
        cached_features_file = os.path.join(
            directory, model_type + "_cached_lm_" + str(block_size)
        )

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            for _, row in df.iterrows():
                conv = construct_conv(row, tokenizer)
                self.examples.append(conv)

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                print(
                    "longest feature (should be 325?)", len(max(self.examples, key=len))
                )
                print(
                    "numer of examples (with 5 turns, should be 67533)",
                    len(self.examples),
                )
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


def load_and_cache_examples(tokenizer, df_trn):
    return ConversationDataset(tokenizer, df_trn)


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

# Prepare optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)

train_dataset = load_and_cache_examples(tokenizer, trn_df)

# train_batch_size = 1 * max(1, args.n_gpu)


def collate(examples: List[torch.Tensor]):
    if tokenizer._pad_token is None:
        return pad_sequence(examples, batch_first=True)
    return pad_sequence(
        examples, batch_first=True, padding_value=tokenizer.pad_token_id
    )


train_sampler = DistributedSampler(train_dataset)
train_dataloader = DataLoader(
    train_dataset, collate_fn=collate, batch_size=1, sampler=train_sampler
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataset)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

train_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, model, optimizer
)

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        inputs, labels = (batch, batch)
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": lr_scheduler.state_dict(),
    },
    "/scratch/project_2001403/poyhnent/results/gpt_rpg_gptneo2.7b.pt",
)
