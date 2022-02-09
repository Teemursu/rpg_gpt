import pandas as pd

df = pd.read_csv("data\persuasion_dialogue_dataset.csv")
print(df["is_persuasion"].value_counts())
