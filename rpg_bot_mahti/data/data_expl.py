import json
import re
import pandas as pd


input_file = "data/NLG_RPG.jsonl"
df = pd.read_json(input_file, lines=True)

df["prompt"] = df["prompt"].astype(str)
df["prompt"] = df["prompt"].str.replace(r"NPC:", "", regex=True)
df["prompt"] = df["prompt"].str.replace(r"Player:", "", regex=True)
print(df["prompt"].apply(lambda x: len(x.split())).sum())
