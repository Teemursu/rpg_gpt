import torch
from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.utils import logging

logging.set_verbosity_info()

tokenizer = AutoTokenizer.from_pretrained(
    "model_new/gpt-neo-2.7b",
    # padding_side="right",
)
model = AutoModelWithLMHead.from_pretrained("model_new/gpt-neo-2.7b")

# Let's chat for 5 lines
for step in range(10):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(
        input(">> Player:") + tokenizer.eos_token,
        return_tensors="pt",
    )
    # print(new_user_input_ids)

    # append the new user input tokens to the chat history
    bot_input_ids = (
        torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        if step > 0
        else new_user_input_ids
    )
    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(
        bot_input_ids,
        # model_max_length=10000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=10,
        top_p=1,
        temperature=0.5,
        max_length=2000,
    )

    # pretty print last ouput tokens from bot
    print(
        "RPG-BOT: {}".format(
            tokenizer.decode(
                chat_history_ids[:, bot_input_ids.shape[-1] :][0],
                skip_special_tokens=True,
            )
        )
    )
