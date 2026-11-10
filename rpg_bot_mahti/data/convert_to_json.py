import pandas as pd
import json
import re


def split_dialogue(group):
    # print(group)
    turns = group.dropna().tolist()
    turns = turns[::-1]
    instances = []
    # print("TURNS: _______________________")
    # for i,turn in enumerate(turns):
    #    print(i, turn)
    # print("_______________________________________")
    # print(turns)
    # print(type(turns))
    for i, elem in enumerate(turns):
        if "NPC:" in turns[i]:
            completion = turns[i]
            prompt = " [SEP] ".join(turns[:i])
            if len(prompt) != 0:
                instances.append({"prompt": prompt, "completion": completion})
    # print(("RESULT"))
    # for instance in instances:
    #    print(instance)
    # print(len(instances))
    return instances


def remove_duplicates(instances):
    tuples = [tuple(instance.items()) for instance in instances]
    # print("tuples", tuples)
    unique_tuples = list(set(tuples))
    unique_instances = [dict(t) for t in unique_tuples]
    return unique_instances


def alternate_speakers(dialogue_dict):
    prompt = dialogue_dict["prompt"]
    lines = prompt.split("\n")
    if any("Player: " in line for line in lines):
        return dialogue_dict
    else:
        lines = list(reversed(lines))
        new_prompt = ""
        for i, line in enumerate(lines):
            if i % 2 == 0:
                new_prompt = "Player: " + line[5:] + " [SEP] " + new_prompt
            else:
                new_prompt = "NPC: " + line[5:] + " [SEP] " + new_prompt
        if new_prompt.endswith(" [SEP] "):
            new_prompt = new_prompt[:-1]
        return {"prompt": new_prompt, "completion": dialogue_dict["completion"]}


def clean_instance(instance):
    prompt = instance["prompt"]
    completion = instance["completion"]

    if "NPC: nan" in prompt or "Player: nan" in prompt or "NPC: nan" in completion:
        return False

    if re.search(r"[A-Z]{4,}", prompt) or re.search(r"[A-Z]{4,}", completion):
        return False

    return True


def generate_instances(df):
    instances = []
    # conversation_dict = {}
    # conversation_id = 0
    # current_conversation = []
    for _, row in df.iterrows():
        print(f"Processing row {_}/", len(df))
        row_instances = split_dialogue(row)
        # print("split dialogue:",row_instances)
        # print()
        for instance in row_instances:
            # instance = alternate_speakers(instance)
            if clean_instance(instance):
                instances.append(instance)
    return remove_duplicates(instances)


def save_to_jsonl(instances, file_path, batch_size=10000):
    with open(file_path, "a") as f:
        for i in range(0, len(instances), batch_size):
            batch = instances[i : i + batch_size]
            try:
                f.writelines([json.dumps(instance) + "\n" for instance in batch])
            except AttributeError:
                print("ERROR:", batch)
                exit()
            print(
                f"Batch {i//batch_size + 1} of {len(instances)//batch_size + 1} has been written."
            )


df = pd.read_csv("data\\NLG_clean_with_speakers.csv", sep="\t")
df.drop("game", axis=1, inplace=True)
df.drop("Unnamed: 0", axis=1, inplace=True)

# df = df.loc[[754]]
# df = df.head(50000)
instances = generate_instances(df)
save_to_jsonl(instances, "NLG_RPG.jsonl")
