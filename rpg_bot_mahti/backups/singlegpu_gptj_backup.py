import torch
from transformers import (
    GPTNeoXTokenizerFast,
    GPTNeoXForCausalLM,
    TrainingArguments,
    Trainer,
)
import json
from torch.utils.data import DataLoader, Dataset

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

        # Truncate the prompt from the left side, so that the most recent dialogue turns
        # give context to the completion, rather than the beginning of dialogue.
        input_ids = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
        ).squeeze()
        input_ids = (
            input_ids[-self.max_length :]
            if len(input_ids) > self.max_length
            else input_ids
        )

        # Padding from the left side, same reasoning as truncation happening from left.
        if len(input_ids) < self.max_length:
            padding = self.tokenizer.encode(
                ("[PAD]" * (self.max_length - len(input_ids))), return_tensors="pt"
            ).squeeze(0)
            input_ids = torch.cat((padding, input_ids), dim=0)
        else:
            pass
        target_ids = self.tokenizer.encode(
            completion,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            # padding_side="left",
        ).squeeze()
        return {"input_ids": input_ids, "labels": target_ids}


train_dataset = "data/NLG_RPG.jsonl"
samples = []
with open(train_dataset, encoding="utf8") as f:
    for sample_line in f:
        sample = json.loads(sample_line)
        prompt = sample["prompt"]
        completion = sample["completion"]
        samples.append((prompt, completion))

# Load tokenizer and model
tokenizer = GPTNeoXTokenizerFast.from_pretrained(
    "Eleutherai/Pythia-1B-deduped",
    cache_dir="cached",
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
model = GPTNeoXForCausalLM.from_pretrained(
    "Eleutherai/Pythia-1B-deduped",
    cache_dir="cached",
    # revision="float16",
    # torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).cuda()
model.resize_token_embeddings()
# Define the data loader
dataset = DialogueDataset(samples, tokenizer)

args = TrainingArguments(
    # model_name="gptj_rpg",
    output_dir="results",
    num_train_epochs=3,
    logging_steps=10,
    per_device_train_batch_size=1,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="logs",
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    # data_collator=collate_fn,
)

torch.cuda.empty_cache()
trainer.train()
torch.cuda.empty_cache()
trainer.save_model()
