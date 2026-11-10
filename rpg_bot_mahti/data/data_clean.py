import json
import re


def clean_instance(instance):
    prompt = instance["prompt"]
    completion = instance["completion"]

    if "nan" in prompt or "nan" in completion:
        return False

    if re.search(r"[A-Z]{4,}", prompt) or re.search(r"[A-Z]{4,}", completion):
        return False

    return True


def clean_data(input_file, output_file):
    with open(input_file, "r") as f_in:
        with open(output_file, "w") as f_out:
            for line in f_in:
                instance = json.loads(line)
                if clean_instance(instance):
                    f_out.write(json.dumps(instance) + "\n")


input_file = "data/NLG_RPG.jsonl"
output_file = "data/NLG_RPG_cleaned.jsonl"

clean_data(input_file, output_file)
