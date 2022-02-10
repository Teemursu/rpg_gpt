import pandas as pd
import string
import re
import numpy as np

langs = ["en", "de", "fr", "it", "es"]
df = pd.read_csv("data\persuasion_dialogue_dataset.csv")


def clean_source(source):
    source = [re.sub("\[[^>]+\]", "", s) for s in source]
    source = [re.sub("\{[^>]+\}", "", s) for s in source]
    source = [re.sub("\{\}", "", s) for s in source]
    source = [re.sub("\[\]", "", s) for s in source]
    source = [str(s).replace("\r", "") for s in source]
    source = [str(s).replace("&apos;", "'") for s in source]
    source = [str(s).replace("\n", "") for s in source]
    source = [str(s).replace("&quot;", '"') for s in source]
    source = [str(s).replace("quot;", '"') for s in source]
    source = [str(s).replace("lt;StartAction", "") for s in source]
    source = [str(s).replace("PERSUASION : ", "") for s in source]
    source = [str(s).replace("&lt;", "") for s in source]
    source = [str(s).replace("&gt;", "") for s in source]
    source = [str(s).replace("</string", "") for s in source]
    source = [str(s).replace(">", "") for s in source]
    source = [str(s).replace("<", "") for s in source]
    source = [str(s).replace("StartCheck/Start", "") for s in source]

    source = [s.rstrip() for s in source]
    source = [x for x in source if x]
    source = [s.lstrip(string.punctuation + string.whitespace) for s in source]
    source = [str(s).replace(">", "") for s in source]
    if len(source) > 0:
        return source[0]
    if len(source) == 0:
        return ""


print(df["is_persuasion"].value_counts())
print(len(df))
df = df.dropna(subset=["sid", "en", "de", "es", "it", "fr"])
print("length after dropna", len(df))
print(df["is_persuasion"].value_counts())


for lang in langs:
    for i, row in df.iterrows():
        try:
            df.at[i, lang] = clean_source([row[lang]])
        except TypeError:
            pass

        if type(row["is_persuasion"]) == str:
            df.at[i, "is_persuasion"] = True
        if type(row["is_persuasion"]) == float:
            df.at[i, "is_persuasion"] = False

        df.at[i, "sid"] = int(row["sid"])


print(df.columns)
print(df["is_persuasion"].value_counts())
print(df.head())

digra_df = df.drop(["filename", "id", "speaker", "replies"], 1)
digra_df.to_csv("digra.csv")
