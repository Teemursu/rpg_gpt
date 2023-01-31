import pandas as pd
import json


def split_dialogue(row):
    turns = []
    for col in row.index:
        if isinstance(row[col], str):
            turns.append(row[col])
    turns = turns[::-1]
    # print(turns)
    instances = []
    for i in range(len(turns)):
        if "NPC:" in turns[i]:
            completion = turns[i]
            if i >= 2:
                prompt = turns[i - 2] + "\n" + turns[i - 1]
                instances.append({"prompt": prompt, "completion": completion})
            elif i >= 1:
                prompt = turns[i - 1]
                instances.append({"prompt": prompt, "completion": completion})
    return instances


# def complete_instances(instances, n=2):
#    for i, instance in enumerate(instances):
#        if not instance["prompt"]:
#            instance["prompt"] = "\n".join(
#                [x["completion"] for x in instances[i - n : i][::-1]]
#            )
#    return instances


def remove_duplicates(instances):
    tuples = [tuple(instance.items()) for instance in instances]
    unique_tuples = list(set(tuples))
    unique_instances = [dict(t) for t in unique_tuples]
    return unique_instances


def generate_instances(df):
    instances = []
    conversation_dict = {}
    conversation_id = 0
    for _, row in df.iterrows():
        print(f"Processing row {_}")
        row_instances = split_dialogue(row)
        for instance in row_instances:
            if len(instance["prompt"].split("\n")) == 1:
                conversation_id += 1
                conversation_dict[conversation_id] = []
            conversation_dict[conversation_id].append(instance)
    print("Processing conversation_dict")
    for conversation_id, conversation in conversation_dict.items():
        for i in range(len(conversation)):
            instance = conversation[i]
            if len(instance["prompt"].split("\n")) == 1:
                if i > 0:
                    prev_instance = conversation[i - 1]
                    instance["prompt"] = (
                        prev_instance["prompt"] + "\n" + prev_instance["completion"]
                    )
                if i < len(conversation) - 1:
                    next_instance = conversation[i + 1]
                    instance["prompt"] = (
                        instance["prompt"] + "\n" + next_instance["prompt"]
                    )
        instances.extend(conversation)
    # print("Removing duplicates")
    return instances


def save_to_jsonl(instances, file_path, batch_size=1000):
    with open(file_path, "a") as f:
        for i in range(0, len(instances), batch_size):
            print(
                f"Starting to write batch {i//batch_size + 1} of {len(instances)//batch_size + 1} "
            )
            batch = instances[i : i + batch_size]
            f.writelines([json.dumps(instance) + "\n" for instance in batch])
            print(
                f"Batch {i//batch_size + 1} of {len(instances)//batch_size + 1} has been written."
            )


df = pd.read_csv("data\\NLG_clean_with_speakers.csv", sep="\t")
df.drop("game", axis=1, inplace=True)

instances = generate_instances(df)
save_to_jsonl(instances, "data\\NLG_RPG_withduplicates.jsonl")
