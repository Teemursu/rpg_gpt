import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("data/NLG_clean.csv", sep="\t")
print(df.shape)
notnull = df.count(axis=0)
print(notnull)
print()
dfs = np.array_split(df, 6, axis=1)
df = pd.concat(dfs, axis=0, ignore_index=True)
print(df.shape)

notnull = df.count(axis=0)
print(notnull)
notnull.to_csv("counts_with6splits.csv")
