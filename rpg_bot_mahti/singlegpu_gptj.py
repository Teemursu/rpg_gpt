import json
import torch
from torch.utils.data import Dataset

from transformers import (
    GPT2Tokenizer,
    GPTJForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


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

        # Tokenize input (prompt)
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

        input_ids = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        #print(input_ids)
        # Tokenize label (completion)
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        
        target_ids = self.tokenizer.encode(
            completion,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        # Truncation
        # Truncate the prompt from the left side.
        # if len(target_ids) > self.max_length - 1:
        #    target_ids = target_ids[: self.max_length - 1]
        # if len(input_ids) > self.max_length - 1:
        #    input_ids = input_ids[-self.max_length + 1 :]

        # input_ids = input_ids.squeeze()
        # target_ids = target_ids.squeeze()
        # Return .long() to prevent token IDs converting to scientific numbers
        # this shouldn't change the token IDs
        # print(input_ids)
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

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(
    "EleutherAI/gpt-j-6B",
    cache_dir="cached",
)

tokenizer.add_special_tokens(
    {
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "additional_special_tokens": [
            "NPC:",
            "Player:",
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
    cache_dir="cached",
    revision="float16",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model.config.use_cache = False
model.resize_token_embeddings(len(tokenizer))
# samples = samples[427:428]
# Create Dataset and Loader
dataset = DialogueDataset(samples, tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

args = TrainingArguments(
    output_dir="results",
    num_train_epochs=20,
    logging_steps=10000,
    per_device_train_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="logs",
    save_strategy="steps",
    save_steps=5000,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="adafactor",
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

torch.cuda.empty_cache()
trainer.train()
torch.cuda.empty_cache()
trainer.save_model()
