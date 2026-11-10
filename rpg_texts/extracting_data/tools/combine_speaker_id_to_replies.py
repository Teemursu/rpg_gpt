import pandas as pd

poe1_ids = pd.read_csv(
    "D:/Code/Data/rpg_texts/data/data/only_ids/poe1_dataset_only_ids.csv", sep="\t"
)
poe2_ids = pd.read_csv(
    "D:/Code/Data/rpg_texts/data/data/only_ids/poe2_dataset_only_ids.csv", sep="\t"
)
tyranny_ids = pd.read_csv(
    "D:/Code/Data/rpg_texts/data/data/only_ids/tyranny_dataset_only_ids.csv", sep="\t"
)


poe1_df = pd.read_csv("D:/Code/Data/rpg_texts/data/data/poe1_dataset.csv", sep="\t")
poe2_df = pd.read_csv("D:/Code/Data/rpg_texts/data/data/poe2_dataset.csv", sep="\t")
tyranny_df = pd.read_csv(
    "D:/Code/Data/rpg_texts/data/data/tyranny_dataset.csv", sep="\t"
)

poe1_df.fillna("", inplace=True)
poe2_df.fillna("", inplace=True)
tyranny_df.fillna("", inplace=True)

for i, row in poe1_ids.iterrows():
    if type(row["speaker"]) != float:
        filename = row["filename"]
        idx = row["id"]
        sid = row["sid"]
        replies = row["replies"]
        entries = row["entries"]
        for i2, row2 in poe1_df.iterrows():
            if poe1_df["filename"][i2] == filename:
                # if poe1_df["id"][i2] == idx:
                # if poe1_df["sid"][i2] == sid:
                if poe1_df["replies"][i2] == replies:
                    if poe1_df["entries"][i2] == entries:
                        # poe1_df.loc[i2, "speaker"] = row["speaker"]
                        poe1_df["speaker"].iloc[i2] = row["speaker"]
                        print(poe1_df.iloc[i2])
                        # print(row2)

poe1_df.to_csv("poe1_dataset_with_speakers.csv", sep="\t")

for i, row in poe2_ids.iterrows():
    filename = row["filename"]
    idx = row["id"]
    sid = row["sid"]
    replies = row["replies"]
    entries = row["entries"]
    for i2, row2 in poe2_df.iterrows():
        if poe2_df["filename"][i2] == filename:
            # if poe2_df["id"][i2] == idx:
            # if poe2_df["sid"][i2] == sid:
            if poe2_df["replies"][i2] == replies:
                if poe2_df["entries"][i2] == entries:
                    poe2_df["speaker"].iloc[i2] = row["speaker"]
                    print(poe2_df.iloc[i2])

poe2_df.to_csv("poe2_dataset_with_speakers.csv", sep="\t")

for i, row in tyranny_ids.iterrows():
    filename = row["filename"]
    idx = row["id"]
    sid = row["sid"]
    replies = row["replies"]
    entries = row["entries"]
    for i2, row2 in tyranny_df.iterrows():
        if tyranny_df["filename"][i2] == filename:
            # if tyranny_df["id"][i2] == idx:
            # if tyranny_df["sid"][i2] == sid:
            if tyranny_df["replies"][i2] == replies:
                if tyranny_df["entries"][i2] == entries:
                    tyranny_df["speaker"].iloc[i2] = row["speaker"]
                    print(tyranny_df.iloc[i2])

tyranny_df.to_csv("tyranny_dataset_with_speakers.csv", sep="\t")
