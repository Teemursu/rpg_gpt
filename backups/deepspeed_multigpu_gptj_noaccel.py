import logging
import os

import torch

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
df = df.dropna()

# trn_df, val_df = train_test_split(df, test_size=0.1)
trn_df = df.head(1)
# trn_df.head()


def construct_conv(row, tokenizer, eos=True):
    flatten = lambda l: [item for sublist in l for item in sublist]
    conv = list(
        reversed([tokenizer.encode(str(x)) + [tokenizer.eos_token_id] for x in row])
    )
    conv = flatten(conv)
    return conv


class ConversationDataset(Dataset):
    def __init__(self, tokenizer: tokenizer, df, block_size=35):
        overwrite_cache = True
        # block_size = block_size #- (
        # tokenizer.model_max_length - tokenizer.max_len_single_sentence
        # )

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
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


def load_and_cache_examples(tokenizer, df_trn):
    return ConversationDataset(tokenizer, df_trn)


train_dataset = load_and_cache_examples(tokenizer, trn_df)
torch.cuda.empty_cache()
# print(examples)
# Prepare training arguments


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
