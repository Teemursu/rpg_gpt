import pandas as pd
import re


def preprocess_df(df):
    # Remove any row that contains a cell that contains something other than a string
    # df = df[df.applymap(lambda x: isinstance(x, str)).all(1)]
    # df = df.drop_duplicates()
    # df.drop("game", axis=1, inplace=True)
    # df = df[df.columns[: list(df.columns).index("context/3") + 1]]
    # df = df[df.applymap(lambda x: isinstance(x, str)).all(1)]
    # remove rows with consecutive capital letters e.g. SYSTEM CONSOLE
    df = df[
        ~df.applymap(lambda x: type(x) == str and bool(re.match(r"\b[A-Z]{4,}\b", x)))
    ]

    return df


def read_and_preprocess_csv(filepath):
    df = pd.read_csv(filepath, sep="\t")
    # df = df.dropna()
    df = preprocess_df(df)
    df.to_csv("NLG_RPG.csv", index=False)


# Example usage
read_and_preprocess_csv("data\\NLG_clean_with_speakers.csv")
