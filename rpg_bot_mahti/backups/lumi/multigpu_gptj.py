import torch
from transformers import (
    GPT2Tokenizer,
    GPTJForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import json
from torch.utils.data import Dataset, DataLoader
import argparse

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
        prompt = self.instances[idx][0] + " => "
        completion = self.instances[idx][1]
        knowledge = self.instances[idx][2]
        # Add EOS token
        # completion += self.tokenizer.eos_token
        # prompt += self.tokenizer.eos_token

        # Padding and truncation for the input on left
        # because we want the most "recent" dialogue/context
        # tokenizer.padding_side = "left"
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
        # tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"

        # Tokenize label (completion)
        target_ids = self.tokenizer.encode(
            completion,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        ).squeeze(0)
        # print("input:", prompt, input_ids)
        # print("output:", completion, target_ids)
        return {"input_ids": input_ids.long(), "labels": target_ids.long()}


train_dataset = "data/NLG_RPG.jsonl"
samples = []
with open(train_dataset, encoding="utf8") as f:
    for sample_line in f:
        sample = json.loads(sample_line)
        prompt = sample["prompt"]
        completion = sample["completion"]
        samples.append((prompt, completion))

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(
    "EleutherAI/gpt-j-6B",
    # return_token_type_ids=False
    # mlm=False,
    # max_length=100,
    # truncation=True,
    # padding="max_length",
    # bos_token="<startoftext>",
    # eos_token="<endoftext>",
)

tokenizer.add_special_tokens(
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
model = GPTJForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B",
    # revision="float16",
    # torch_dtype=torch.float16
)
# )
model.resize_token_embeddings(len(tokenizer))
# model.to(torch.float16)
# Define the data loader
dataset = DialogueDataset(samples, tokenizer)
# train_sampler = DistributedSampler(dataset)
# data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
# model = accelerator.prepare(model)
training_args = TrainingArguments(
    # model_name="gptj_rpg",
    output_dir="results",
    num_train_epochs=3,
    logging_steps=10,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    eval_steps=10000,
    logging_dir="logs",
    save_strategy="epoch",
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
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
