import pandas as pd
import ast
import numpy as np


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
    df.persuasion = df.persuasion.fillna("False")
    # df.persuasion = df.persuasion.apply(ast.literal_eval)
    for i, row in df.iterrows():
        if type(row["persuasion"]) != bool:
            if "[" not in row["persuasion"]:
                df.at[i, "persuasion"] = str(row["persuasion"])
            elif row["persuasion"] == "[]":
                row["persuasion"] = "False"
            else:
                print()
                print(row["persuasion"])
                print(type(row["persuasion"]))
                df.at[i, "persuasion"] = ast.literal_eval(row["persuasion"])
                print(type(row["persuasion"]))

    return df


kotor_nwn_df = pd.read_csv("data\persuasion_dialogue_dataset.csv")
kotor_nwn_df = kotor_nwn_df.rename(columns={"is_persuasion": "persuasion"})
poe2_df = pd.read_csv("data\poe2_dataset.csv", sep="\t")
poe1_df = pd.read_csv("data\poe1_dataset.csv", sep="\t")
tyranny_df = pd.read_csv("data\\tyranny_dataset.csv", sep="\t")

df_list = [kotor_nwn_df, poe2_df, poe1_df, tyranny_df]
df_list = [rm_unnecessary_columns(df) for df in df_list]

df = pd.concat([df for df in df_list])
df = df.dropna(subset=["en"])
df = unify_persuasion(df)
print(df["persuasion"].value_counts())
