import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("nlg_data_10_turns_all_data.csv", sep="\t")
df = df.drop_duplicates()

print(df.shape)
df = df[df.columns[: list(df.columns).index("context/1") + 3]]
print(df.shape)
df = df.drop_duplicates(subset="context/0")
print(df.shape)
df = df.drop_duplicates(subset="context/1")
print(df.shape)
df = df.drop_duplicates(subset="context/2")
print(df.shape)
df = df.drop_duplicates(subset="context/3")
print(df.shape)
df = df.drop_duplicates(subset="response")
df = df.dropna()
print(df.shape)
