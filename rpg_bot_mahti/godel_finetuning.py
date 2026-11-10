import json
import os
import torch
from torch.utils.data import Dataset

# Smoke-test overrides (added 2026; defaults reproduce the thesis run):
#   RPG_GPT_BASE_MODEL  base model           (microsoft/GODEL-v1_1-large-seq2seq)
#   RPG_GPT_DATA        prompt/completion jsonl (data/NLG_RPG.jsonl)
#   RPG_GPT_EPOCHS, RPG_GPT_BATCH_SIZE, RPG_GPT_MAX_STEPS, RPG_GPT_OUTPUT_DIR
BASE_MODEL = os.environ.get("RPG_GPT_BASE_MODEL", "microsoft/GODEL-v1_1-large-seq2seq")

from transformers import (
    # GPT2Tokenizer,
    AutoTokenizer,
    # GPTJForCausalLM,
    GPTNeoXForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)


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
        self.tokenizer.truncation_side = "left"

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
        self.tokenizer.truncation_side = "right"

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


# Load data from file
train_dataset = os.environ.get("RPG_GPT_DATA", "data/NLG_RPG.jsonl")
samples = []
with open(train_dataset, encoding="utf8") as f:
    for sample_line in f:
        sample = json.loads(sample_line)
        prompt = sample["prompt"]
        completion = sample["completion"]
        samples.append((prompt, completion, ""))

# Load tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained(
#    "EleutherAI/gpt-j-6B",
#    cache_dir="cached",
# )
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir="cached")

# tokenizer.pad_token = tokenizer.eos_token
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
# model = GPTJForCausalLM.from_pretrained(
model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    # "EleutherAI/pythia-2.8b",
    cache_dir="cached",
    # revision="float16",
    # torch_dtype=torch.float16,
    # task=["summarization"],
    # low_cpu_mem_usage=True,
)
model.config.use_cache = False
model.resize_token_embeddings(len(tokenizer))

# Create Dataset and Loader
dataset = DialogueDataset(samples, tokenizer)
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=tokenizer.pad_token_id,  # mlm=False
)

args = TrainingArguments(
    output_dir=os.environ.get("RPG_GPT_OUTPUT_DIR", "results"),
    num_train_epochs=int(os.environ.get("RPG_GPT_EPOCHS", 20)),
    max_steps=int(os.environ.get("RPG_GPT_MAX_STEPS", -1)),
    logging_steps=1000,
    per_device_train_batch_size=int(os.environ.get("RPG_GPT_BATCH_SIZE", 128)),
    warmup_steps=100,
    weight_decay=0.01,
    # logging_dir="logs",  # removed in transformers 5; original run logged to logs/
    save_strategy="epoch",
    # save_steps=1001,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    optim="adamw_torch",
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=data_collator,
    processing_class=tokenizer,  # was tokenizer= (removed in transformers 5)
)

torch.cuda.empty_cache()
trainer.train(resume_from_checkpoint=False)
torch.cuda.empty_cache()
trainer.save_model()
