import pandas as pd
import ast
import numpy as np
import re
import string


def clean_source(source):
    # source = [str(s)[:-1] if s[-1] == '"' else s for s in source]
    source = [str(s) for s in source]

    source = [re.sub("\{[^>]+\}", "", s) for s in source]
    source = [re.sub("\([^>]+\)", "", s) for s in source]
    source = [re.sub("<[^>]+>", "", s) for s in source]

    source = [re.sub("\{\}", "", s) for s in source]
    # source = [re.sub("\[\]", "", s) if s[:-1] != "]" else s for s in source ]
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
    source = [re.sub(r"\[url=glossary:.*?\]", "", s) for s in source]
    source = [str(s).replace("</i>", "") for s in source]
    source = [str(s).replace("<i>", "") for s in source]

    source = [str(s).replace("<ispeech>", "") for s in source]

    source = [str(s).replace(r"[/url]", "") for s in source]
    source = [str(s).replace("StartCheck/Start", "") for s in source]
    source = [str(s).replace("StartAction/Start", "") for s in source]
    source = [str(s).replace("StartCheckEingebung/Start", "") for s in source]
    source = [str(s).replace("StartActionIntuizione/Start", "") for s in source]
    source = [str(s).replace("StartCheckEingebung/Start", "") for s in source]
    source = [str(s).replace("StartCheckEingebung/Start", "") for s in source]
    source = [str(s).replace("StartCheckEingebung/Start", "") for s in source]
    source = [str(s).replace(">", "") for s in source]
    source = [str(s).replace("<", "") for s in source]
    source = [str(s).replace("[Player Name]", "<|playername|>") for s in source]
    source = [str(s).replace("[Player Ship]", "the ship") for s in source]
    source = [str(s).replace("[Player Culture]", "<|playerculture|>") for s in source]
    source = [str(s).replace("[Player Race]", "<|playerrace|>") for s in source]

    source = [str(s).replace("FirstName", "<|playername|>") for s in source]
    # source = [s.rstrip() for s in source]
    source = [x for x in source if x]
    # source = [s.lstrip(string.punctuation + string.whitespace) for s in source]
    source = [str(s).replace("/ispeech", "") for s in source]
    source = [re.sub("\[[^>]+\]", "", str(s)) for s in source]

    if len(source) > 0:
        return source[0]
    if len(source) == 0:
        return np.nan


def rm_unnecessary_columns(df):
    col_list = [
        "game",
        "filename",
        "id",
        "sid",
        "replies",
        # "entries",
        "persuasion",
        "en",
    ]
    return df.drop(columns=[col for col in df if col not in col_list])


def unify_persuasion(df):
    df.persuasion = df.persuasion.fillna(False)
    # df.persuasion = df.persuasion.apply(ast.literal_eval)
    for i, row in df.iterrows():
        if type(row["persuasion"]) != bool:

            if "[" not in row["persuasion"]:
                # df.at[i, "persuasion"] = str(row["persuasion"])
                df.at[i, "persuasion"] = str(df.at[i, "persuasion"])
            elif row["persuasion"] == "[]":
                df.at[i, "persuasion"] = False
            else:
                # df.at[i, "persuasion"] = ast.literal_eval(row["persuasion"])
                row = row["persuasion"].strip("][").split(", ")
                row = row[0]
                row = re.sub("'", "", row)
                df.at[i, "persuasion"] = row

            try:
                test = row["persuasion"].__contains__("Persuade")
                test2 = row["persuasion"].__contains__("/")
                test3 = row["persuasion"].__contains__("<StartAction>")
                test5 = row["persuasion"].__contains__("_Handed")
                test6 = row["persuasion"].__contains__("Spell_")
                if test and test2:
                    df.at[i, "persuasion"] = "Persuade"
                if test3:
                    df.at[i, "persuasion"] = "Persuade"
                if row["persuasion"][0].isdigit():
                    df.at[i, "persuasion"] = False
                if test5:
                    df.at[i, "persuasion"] = "Swords"
                if test6:
                    df.at[i, "persuasion"] = "Magic"
                if "}TreatInjury" in row["persuasion"]:
                    df.at[i, "persuasion"] = "TreatInjury"
                if "}Computer" in row["persuasion"]:
                    df.at[i, "persuasion"] = "Computer"
                if "}Repair" in row["persuasion"]:
                    df.at[i, "persuasion"] = "Repair"
                if "TrainingQuestneedmultipleflagstatesinonenode." in row["persuasion"]:
                    df.at[i, "persuasion"] = False
                if "ght>Insight" in row["persuasion"]:
                    df.at[i, "persuasion"] = "Insight"

                # Combine tags
                if "ComputerSkill" in row["persuasion"]:
                    df.at[i, "persuasion"] = "Computer"
                if "Strength" in row["persuasion"]:
                    df.at[i, "persuasion"] = "Athletics"
                if df.at[i, "persuasion"] == "Intellect":
                    df.at[i, "persuasion"] = "Intelligence"
            except TypeError:
                pass

    return df


def get_dialogue_only(source):
    # print(source)
    # print("turns into")
    if type(source) == float:
        return np.nan
    if r'"' not in source:
        source = ""
    m = re.findall('"([^"]*)"', source)
    source = " ".join(m)
    # print(source)
    if len(source) == 0:
        return np.nan
    return source


kotor_nwn_df = pd.read_csv("persuasion_dialogue_dataset.csv")
kotor_nwn_df = kotor_nwn_df.rename(columns={"is_persuasion": "persuasion"})
poe2_df = pd.read_csv("data\poe2_dataset.csv", sep="\t")
poe1_df = pd.read_csv("data\poe1_dataset.csv", sep="\t")
tyranny_df = pd.read_csv("data\\tyranny_dataset.csv", sep="\t")

df_list = [kotor_nwn_df, poe2_df, poe1_df, tyranny_df]
df_list = [rm_unnecessary_columns(df) for df in df_list]

df = pd.concat(df_list, ignore_index=True)
df = df.dropna(subset=["en"])
df = unify_persuasion(df)
for i, row in df.iterrows():
    df.at[i, "en"] = clean_source([row["en"]])
    if row["game"] == "poe2" or row["game"] == "poe1" or row["game"] == "tyranny":
        # print(row["game"])
        df.at[i, "en"] = get_dialogue_only(df.at[i, "en"])

df.loc[df.persuasion == "Intellect"] = "Intelligence"
df.loc[df.persuasion == "Spellcraft"] = "Magic"
# df.loc[df.persuasion == "Awareness"] = "Insight"
rows = df["en"][(df.persuasion == "Constitution") & (df.en != float)]
rows = rows.dropna()
print(rows)
print()
print(df["persuasion"].value_counts())
df = df.dropna(subset=["en"])
df.to_csv(
    "persuasion_dataset_w_quotes_and_squarebrackets_and_more_skills.csv",
    encoding="utf-8",
    sep="\t",
)
# 319584
#
