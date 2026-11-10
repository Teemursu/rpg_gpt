from cmath import nan
import pandas as pd
import numpy as np

df = pd.read_csv("data\\NLG_clean_with_speakers.csv", sep="\t")
# df = df.drop(labels=["Unnamed: 0", "Unnamed: 0.1"], axis=1)
cols = list(df.columns.values)  # Make a list of all of the columns in the df
cols.pop(cols.index("game"))  # Remove b from list
df = df[cols + ["game"]]  # Create new dataframe with columns in the order you want
print(df.shape)
new_df = pd.DataFrame(
    columns=[
        "response",
        "context/0",
        "context/1",
        "context/2",
        "context/3",
        # "context/4",
        # "context/5",
        # "context/6",
        # "context/7",
        # "context/8",
        "game",
    ]
)


def divide_chunks(l, n):

    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


print(df.columns)
for i, row in df.iterrows():
    row_lst = list(row)
    row_lst = row_lst[:-1]
    row_lst = [x for x in row_lst if type(x) != float]
    row_lsts = list(divide_chunks(row_lst, 5))
    for context_list in row_lsts:
        while len(context_list) < 6:
            context_list.append(np.nan)
        new_df.loc[len(new_df)] = context_list
        new_df["game"].loc[len(new_df) - 1] = str(row["game"])
        print(new_df.loc[len(new_df) - 1])
        print()

print(new_df)
print(new_df.shape)
new_df.to_csv("nlg_data_5_turns_all_data_with_speakers.csv", sep="\t")
