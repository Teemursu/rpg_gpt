import torch
from transformers import (
    AutoTokenizer,
    GPTNeoXForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import json
from torch.utils.data import Dataset, DataLoader
import argparse

import deepspeed

deepspeed.init_distributed()
# torch.cuda.set_device(torch.cuda.current_device())
torch.cuda.manual_seed_all(42)


class DialogueDataset(Dataset):
    def __init__(self, instances, tokenizer):
        self.instances = instances
        self.tokenizer = tokenizer
        self.max_length = 256

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        prompt = self.instances[idx][0]
        completion = self.instances[idx][1]

        # Add EOS token
        completion += self.tokenizer.eos_token
        prompt += self.tokenizer.eos_token

        # Padding and truncation for the input on left
        # because we want the most "recent" dialogue/context
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

        # Tokenize input (prompt)
        input_ids = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
            padding="max_length",
            # padding=True,
            truncation=True,
            max_length=self.max_length,
        ).squeeze(0)

        # Normal padding/truncation for output/labels (completion)
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"

        # Tokenize label (completion)
        target_ids = self.tokenizer.encode(
            completion,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        ).squeeze(0)
        print("input:", prompt, input_ids)
        print("output:", completion, target_ids)
        return {"input_ids": input_ids.long(), "labels": target_ids.long()}


# Load data from file
train_dataset = "data/NLG_RPG.jsonl"
samples = []
with open(train_dataset, encoding="utf8") as f:
    for sample_line in f:
        sample = json.loads(sample_line)
        prompt = sample["prompt"]
        completion = sample["completion"]
        samples.append((prompt, completion))

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-2.8b-deduped", cache_dir="cached"
)
# return_token_type_ids=False
# mlm=False,
# max_length=100,
# truncation=True,
# padding="max_length",
# bos_token="<startoftext>",
# eos_token="<endoftext>",

tokenizer.add_special_tokens(
    {
        "sep_token": "[SEP]",
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
model = GPTNeoXForCausalLM.from_pretrained(
    "EleutherAI/pythia-2.8b-deduped",
    # "EleutherAI/pythia-2.8b",
    cache_dir="cached",
    # revision="float16",
    # torch_dtype=torch.float16,
    # low_cpu_mem_usage=True,
)
# )
model.config.use_cache = False
model.resize_token_embeddings(len(tokenizer))
# model.to(torch.float16)
# Define the data loader
dataset = DialogueDataset(samples, tokenizer)
# train_sampler = DistributedSampler(dataset)
# data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
# model = accelerator.prepare(model)
training_args = TrainingArguments(
    output_dir="results",
    num_train_epochs=20,
    logging_steps=10000,
    per_device_train_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="logs",
    save_strategy="steps",
    save_steps=1001,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="adafactor",
    deepspeed="/scratch/project_2001403/poyhnent/configs/ds_config.json",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    # data_collator=data_collator,
    tokenizer=tokenizer,
    # plugins="fsdp",
)
torch.cuda.empty_cache()
trainer.train()
torch.cuda.empty_cache()
trainer.save_model()
