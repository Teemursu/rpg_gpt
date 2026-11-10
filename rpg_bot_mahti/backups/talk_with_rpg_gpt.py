import torch
from transformers import (
    GPTJForCausalLM,
    GPT2Tokenizer,
    GPTNeoXForCausalLM,
    AutoTokenizer,
)
from transformers.utils import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.set_verbosity_info()

tokenizer = AutoTokenizer.from_pretrained(
    "results/checkpoint-1001"
    # pad_token="[PAD]"
)
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

model = GPTJForCausalLM.from_pretrained(
    "results/checkpoint-1001",
    # revision="float16",
    # torch_dtype=torch.float16,
    # low_cpu_mem_usage=True,
)
model.cuda()
# Let's chat for 5 lines
for step in range(5):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(
        input(">> Player:") + tokenizer.eos_token, return_tensors="pt"
    ).cuda()
    # print(new_user_input_ids)

    # append the new user input tokens to the chat history
    bot_input_ids = (
        torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        if step > 0
        else new_user_input_ids
    ).cuda()
    print(bot_input_ids)
    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(
        bot_input_ids,
        # model_max_length=10000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=False,
        top_k=3,
        top_p=0.9,
        temperature=1,
        max_length=200,
    ).cuda()
    # print(chat_history_ids)
    # pretty print last ouput tokens from bot
    print(
        "RPG-BOT: {}".format(
            tokenizer.decode(
                chat_history_ids[:, bot_input_ids.shape[-1] :][0],
                skip_special_tokens=True,
            )
        )
    )
