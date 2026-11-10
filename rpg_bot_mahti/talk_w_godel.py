from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

modelname = "results/checkpoint-13236"
tokenizer = AutoTokenizer.from_pretrained(modelname)
model = AutoModelForSeq2SeqLM.from_pretrained(modelname)


# Instruction for a chitchat task
# npc_personality = input(
#    "Which NPC are you talking to? Describe personality, name, etc.\nThey will answer as someone who... "
# )
npc_personality = ""
if npc_personality != "":
    instruction = f"Instruction: given a dialog context, you need give an engaging and relevant answer as {npc_personality}."
else:
    instruction = f"Instruction: given a dialogue context, respond by giving an engaging answer. Do not repeat questions or phrases."
# Leave the knowldge empty
knowledge = ""

dialog = []
# response = generate(instruction, knowledge, dialog)
# print(response)

print()
dialogue = dialog
# eos_token = tokenizer.eos_token
for i in range(50):
    player_input = input("Player: ")
    player_input = f"Player: {player_input}"
    # print(tokenizer(player_input, return_tensors="pt"))
    # print(tokenizer("Player: ", return_tensors="pt"))
    dialogue.append(player_input)
    if knowledge != "":
        knowledge = "[KNOWLEDGE] " + knowledge
    # print(dialogue)
    dialog = " [SEP] ".join(dialogue)
    query = f"{instruction} [CONTEXT] {dialog} => "
    # query = f"{dialog} => "
    # print(query)
    input_ids = tokenizer(f"{query}" + " NPC: ", return_tensors="pt").input_ids
    outputs = model.generate(
        input_ids,
        max_length=256,
        min_length=2,
        # The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        # min_new_tokens=5,
        # If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        top_p=1,
        # pad_token_id=tokenizer.eos_token_id,
        # all ngrams of that size can only occur once.
        no_repeat_ngram_size=1,
        # The size of the candidate set that is used to re-rank for contrastive search
        # top_k=5000,
        # The value used to module the next token probabilities.
        temperature=0.90,
        # typical_p (float, optional, defaults to 1.0) — Local typicality measures how similar the conditional probability of predicting a target token next is to the expected conditional probability of predicting a random token next, given the partial text already generated. If set to float < 1, the smallest set of the most locally typical tokens with probabilities that add up to typical_p or higher are kept for generation. See this paper for more details.
        # typical_p=0.9,
        repetition_penalty=1,
        do_sample=True,
        # prompt_end="[SEP]",
    )
    # print(outputs)
    # print(tokenizer("NPC: ", return_tensors="pt"))
    output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    output = re.sub("<pad>", "", output)
    output = re.sub("</s>", "", output)
    print(output)
    dialogue.append(output)
