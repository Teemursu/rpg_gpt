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

        completion += self.tokenizer.eos_token
        prompt += self.tokenizer.eos_token
        # Truncate the prompt from the left side, so that the most recent dialogue turns
        # give context to the completion, rather than the beginning of dialogue.
        input_ids = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
        ).squeeze()

        # Padding from the left side, same reasoning as truncation happening from left.
        if len(input_ids) < self.max_length:
            padding = self.tokenizer.encode(
                ("[PAD]" * (self.max_length - len(input_ids) - 1)), return_tensors="pt"
            ).squeeze()
            input_ids = torch.cat(
                (
                    padding,
                    input_ids,
                ),
                dim=0,
            )
        else:
            pass
        target_ids = self.tokenizer.encode(
            completion,
            return_tensors="pt",
            padding="max_length",
            truncation=False,
            max_length=self.max_length,
            # padding_side="left",
        ).squeeze()

        if len(target_ids) > self.max_length - 1:
            target_ids = target_ids[: self.max_length - 1]
        if len(input_ids) > self.max_length - 1:
            input_ids = input_ids[-self.max_length + 1 :]

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
    cache_dir="cached",
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
        "sep_token": "\n",
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
    # torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
# )
model.resize_token_embeddings(len(tokenizer))
# model.to(torch.float16)
# Define the data loader
dataset = DialogueDataset(samples, tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
args = TrainingArguments(
    output_dir="results",
    num_train_epochs=3,
    logging_steps=1000,
    per_device_train_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="logs",
    save_strategy="steps",
    save_steps=20000,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    # fp16=True,
    optim="adafactor",
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    # plugins="fsdp",
    # use_cache=False,
)

torch.cuda.empty_cache()
trainer.train()
torch.cuda.empty_cache()
trainer.save_model()
