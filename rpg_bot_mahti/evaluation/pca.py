from scipy.stats import ttest_ind, shapiro, kstest, wilcoxon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import required libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import preprocessing
from factor_analyzer import FactorAnalyzer
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt

from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from sklearn.decomposition import PCA


def histogram(df):
    print()
    plt.hist(df["value"])
    plt.show()


def calc_ttest(df1, df2):
    df1 = df1.reindex(sorted(df1.columns), axis=1).melt()
    df2 = df2.reindex(sorted(df2.columns), axis=1).melt()

    df2 = df2.fillna(df2["value"].median())
    df1 = df1.fillna(df1["value"].median())

    # histogram(df1)
    ##histogram(df2)
    # print("/" * 15)
    print(df1.describe())
    print(df2.describe())
    # print("/" * 15)
    print(
        shapiro(
            df1["value"],
        )
    )
    print(
        shapiro(
            df2["value"],
        )
    )
    print(kstest(df1["value"], "norm"))
    print(kstest(df2["value"], "norm"))
    print("/" * 15)
    print(
        wilcoxon(
            df1["value"],
            df2["value"],
        )
    )


df = pd.read_csv("evaluation/survey_data861.csv", sep=";")

methods = ["game_", "ai_"]
conversations = [
    "laeth",
    "help",
    "teaching",
    "eothas",
    "merchant",
    "teaching",
    "refresh",
    "menzzago",
]
questions = ["releva", "intent", "attent", "drawn"]
# print(df.head(1))


# df = df.loc[df["games_played"] != "None"]
# game_df = df.loc[df["games_played"] == "None"]
##ai_df = df.loc[df["games_played"] != "None"]
# df = df.filter(regex=r"merchant|teaching|eothas")
game_df = df.filter(regex="game_", axis=1)
ai_df = df.filter(regex="ai_", axis=1)
ai_releva_df = ai_df.filter(regex="releva", axis=1)
game_releva_df = game_df.filter(regex="releva", axis=1)
ai_releva_df = pd.Series(ai_releva_df.to_numpy().ravel(order="F"), name="ai_relevance")
game_releva_df = pd.Series(
    game_releva_df.to_numpy().ravel(order="F"), name="game_relevance"
)

ai_intent_df = ai_df.filter(regex="intent", axis=1)
game_intent_df = game_df.filter(regex="intent", axis=1)
ai_intent_df = pd.Series(ai_intent_df.to_numpy().ravel(order="F"), name="ai_intent")
game_intent_df = pd.Series(
    game_intent_df.to_numpy().ravel(order="F"), name="game_intent"
)

ai_attent_df = ai_df.filter(regex="attent", axis=1)
game_attent_df = game_df.filter(regex="attent", axis=1)
ai_attent_df = pd.Series(ai_attent_df.to_numpy().ravel(order="F"), name="ai_attent")
game_attent_df = pd.Series(
    game_attent_df.to_numpy().ravel(order="F"), name="game_attent"
)

ai_drawn_df = ai_df.filter(regex="drawn", axis=1)
game_drawn_df = game_df.filter(regex="drawn", axis=1)
ai_drawn_df = pd.Series(ai_drawn_df.to_numpy().ravel(order="F"), name="ai_drawn")
game_drawn_df = pd.Series(game_drawn_df.to_numpy().ravel(order="F"), name="game_drawn")

# game_df = game_df.filter(regex="drawn", axis=1)
# ai_df = ai_df.filter(regex="drawn", axis=1)

# calc_ttest(game_df, ai_df)
# df = pd.concat([game_df, ai_df], axis=1)
df = pd.concat(
    [
        ai_releva_df,
        ai_intent_df,
        ai_attent_df,
        ai_drawn_df,
        game_releva_df,
        game_intent_df,
        game_attent_df,
        game_drawn_df,
    ],
    axis=1,
)
df.dropna(inplace=True)
print("df info")
# print(df.info())
print(df.corr())
chi_square_value, p_value = calculate_bartlett_sphericity(df)
print("chi:", chi_square_value, "p:", p_value)
from factor_analyzer.factor_analyzer import calculate_kmo

kmo_all, kmo_model = calculate_kmo(df)
print("kmo", kmo_model)

fa = FactorAnalyzer(n_factors=4, rotation="varimax")
fa.fit(df)
loadings = fa.loadings_
ev, v = fa.get_eigenvalues()

xvals = range(1, df.shape[1] + 1)
plt.scatter(xvals, ev)
plt.plot(xvals, ev)
plt.title("Scree plot")
plt.xlabel("Factor")
plt.ylabel("Eigenvalue")
plt.grid()
plt.show()

sns.heatmap(
    data=df.corr(),
    vmin=-1,
    vmax=1,
    center=0,
    # cmap=sns.diverging_palette(20, 220, n=470),
    square=True,
)
##plt.show()
print(pd.DataFrame.from_records(loadings))
print(fa.corr_)
print(fa.rotation_matrix_)
print(fa.get_factor_variance())
# rint(fa.transform(X=df))
