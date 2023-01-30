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
max_length = 10
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
trn_df = df.head(30)
# trn_df.head()


def construct_conv(row, tokenizer, eos=True):
    flatten = lambda l: [item for sublist in l for item in sublist]
    conv = list(
        reversed([tokenizer.encode(str(x)) + [tokenizer.eos_token_id] for x in row])
    )
    conv = flatten(conv)
    return conv


class ConversationDataset(Dataset):
    def __init__(self, tokenizer: tokenizer, df, block_size=512):
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


train_sampler = DistributedSampler(train_dataset)
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
        inputs, labels = (batch, batch)
        outputs = model(inputs, labels=labels)
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
