import torch
from transformers import (
    GPTJForCausalLM,
    GPT2Tokenizer,
)
from transformers.utils import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.set_verbosity_info()

tokenizer = GPT2Tokenizer.from_pretrained(
    # "EleutherAI/gpt-j-6B",
    "results/checkpoint-20000"
)
model = GPTJForCausalLM.from_pretrained(
    # "EleutherAI/gpt-j-6B",
    "results/checkpoint-20000",
    revision="float16",
    torch_dtype=torch.float16,
    # low_cpu_mem_usage=False,
)
model.cuda()
# Let's chat for 5 lines
with torch.no_grad():
    for step in range(10):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        new_user_input_ids = tokenizer(
            input("Player: ") + tokenizer.eos_token,
            return_tensors="pt",
        ).input_ids
        # print(new_user_input_ids)
        print("input", new_user_input_ids)
        # append the new user input tokens to the chat history
        bot_input_ids = (
            torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
            if step > 0
            else new_user_input_ids
        ).cuda()
        # ).long()
        # bot_input_ids = new_user_input_ids.long()
        # tokenizer.padding_side = "left"
        # tokenizer.truncation_side = "right"
        # generated a response while limiting the total chat history to 1000 tokens,
        chat_history_ids = model.generate(
            bot_input_ids,
            # model_max_length=10000,
            pad_token_id=tokenizer.eos_token_id,
            # no_repeat_ngram_size=3,
            # do_sample=True,
            # top_k=5,
            # top_p=0.7,
            temperature=0.8,
            max_length=600,
            # max_new_tokens=600,
        ).cuda()

        # pretty print last ouput tokens from bot
        print(
            "NPC: {}".format(
                tokenizer.decode(
                    chat_history_ids[:, bot_input_ids.shape[-1] :][0],
                    skip_special_tokens=True,
                )
            )
        )
