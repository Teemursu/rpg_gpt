import logging
import os

import torch

from transformers import (
    get_scheduler,
    GPT2Tokenizer,
    GPTNeoForCausalLM,
    Trainer,
    TrainingArguments,
)
import pandas as pd

logger = logging.getLogger(__name__)
from accelerate import Accelerator

from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

import pickle
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm


model_type = "gpt_neo_2.7b"

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(
    "EleutherAI/gpt-neo-2.7B",
    cache_dir="cached",
    mlm=False,
    # max_length=10,
    # truncation=True,
    # padding="max_length"
    # max_len_single_sentence=2
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

# trn_df, val_df = train_test_split(df, test_size=0.1)
trn_df = df.head(1)  # .head(30)
print(trn_df)
trn_df.head()


def construct_conv(row, tokenizer, eos=True):
    flatten = lambda l: [item for sublist in l for item in sublist]
    conv = list(
        reversed([tokenizer.encode(str(x)) + [tokenizer.eos_token_id] for x in row])
    )
    conv = flatten(conv)
    return conv


class ConversationDataset(Dataset):
    def __init__(self, tokenizer: tokenizer, df, block_size=1):
        overwrite_cache = True
        # block_size = block_size #- (
        # tokenizer.model_max_length - tokenizer.max_len_single_sentence
        # )

        directory = "cached"
        cached_features_file = os.path.join(
            directory, model_type + "_cached_lm_" + str(block_size)
        )

        if os.path.exists(cached_features_file) and not overwrite_cache:
            print("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            print("Creating features from dataset file at %s", directory)

            self.examples = []
            for _, row in df.iterrows():
                conv = construct_conv(row, tokenizer)
                self.examples.append(conv)

            print("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


def load_and_cache_examples(tokenizer, df_trn):
    return ConversationDataset(tokenizer, df_trn)


train_dataset = load_and_cache_examples(tokenizer, trn_df)

max_length = 10
truncate = True


def collate(examples: List[torch.Tensor]):
    max_length = 10
    truncate = True
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


train_dataloader = DataLoader(
    train_dataset, collate_fn=collate, batch_size=1, shuffle=True
)
for example in train_dataloader:
    print(example)
    print(torch.contiguous(example))
    print()
for i, example in enumerate(train_dataloader):
    example_len = example.shape[1]
    if example_len > max_length:
        max_length = example_len
        longest_example_index = i
print(
    "Longest  is with length",
    max_length,
)
