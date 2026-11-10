from scipy.stats import ttest_ind, shapiro, kstest, wilcoxon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from decimal import *

warnings.simplefilter(action="ignore", category=FutureWarning)
# Import required libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import preprocessing
from factor_analyzer import FactorAnalyzer
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt


def histogram(df):
    print()
    plt.hist(df["value"])
    plt.show()


def calc_ttest(df1, df2):
    try:
        df1 = df1.reindex(sorted(df1.columns), axis=1).melt()
        df2 = df2.reindex(sorted(df2.columns), axis=1).melt()
        df2 = df2.fillna(df2["value"].median())
        df1 = df1.fillna(df1["value"].median())

        ttest = ttest_ind(df1["value"], df2["value"])

    except:
        df2 = df2.fillna(df2.median())
        df1 = df1.fillna(df1.median())
        ttest = ttest_ind(df1, df2)
    print("/" * 15)
    print("df info")
    print(df1.info())
    # histogram(df1)
    ##histogram(df2)
    # print("/" * 15)
    game_mean = df1.mean()
    ai_mean = df2.mean()
    game_std = df1.std()
    ai_std = df2.std()
    # print("/" * 15)
    ##   print(
    ##      shapiro(
    ##         df1["value"],
    ##3     )
    #  )
    ## print(
    #     shapiro(
    ##         df2["value"],
    #     )
    ## )
    result = [ttest[0], ttest[1], game_mean, game_std, ai_mean, ai_std]
    print("before", result)
    result = [float(Decimal("%.4f" % e)) for e in result]
    og_result = [str(x) for x in result]
    ai_stats = [" +- ".join(og_result[-2:])]
    game_stats = [" +- ".join([og_result[2], og_result[3]])]
    # print("ai:", ai_stats)
    # print("game:", game_stats)
    result = [og_result[0], og_result[1], game_stats[0], ai_stats[0], len(df2)]
    print("AI mean:\t", ai_stats[0])
    print("Game mean:\t", game_stats[0])
    print("p-value:\t", og_result[1])
    return result


og_df = pd.read_csv("evaluation/survey_data861.csv", sep=";")


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

wilcox_df = pd.DataFrame()

###for context in conversations:
context = "all"
if context == "all":
    col_series = []
    df = og_df.filter(regex=r"game_|ai_")

    # df.dropna(inplace=True)
    # df = df.filter(regex=context)

    # print("conv:", context)
    #
    # df = df.astype(int)

    game_df = df.filter(regex="game_", axis=1)
    ai_df = df.filter(regex="ai_", axis=1)

    print("Game VS AI:")
    wilcox_df[context] = calc_ttest(game_df, ai_df)

    ai_releva_df = ai_df.filter(regex="releva", axis=1)
    game_releva_df = game_df.filter(regex="releva", axis=1)
    ai_releva_df = pd.Series(
        ai_releva_df.to_numpy().ravel(order="F"), name="ai_relevance"
    )
    game_releva_df = pd.Series(
        game_releva_df.to_numpy().ravel(order="F"), name="game_relevance"
    )

    print("Game relevance VS AI relevance:")
    wilcox_df[context + "_relevance"] = calc_ttest(game_releva_df, ai_releva_df)

    ai_intent_df = ai_df.filter(regex="intent", axis=1)
    game_intent_df = game_df.filter(regex="intent", axis=1)
    ai_intent_df = pd.Series(ai_intent_df.to_numpy().ravel(order="F"), name="ai_intent")
    game_intent_df = pd.Series(
        game_intent_df.to_numpy().ravel(order="F"), name="game_intent"
    )

    print("Game intent VS AI intent:")
    wilcox_df[context + "_intention"] = calc_ttest(game_intent_df, ai_intent_df)

    ai_attent_df = ai_df.filter(regex="attent", axis=1)
    game_attent_df = game_df.filter(regex="attent", axis=1)
    ai_attent_df = pd.Series(ai_attent_df.to_numpy().ravel(order="F"), name="ai_attent")
    game_attent_df = pd.Series(
        game_attent_df.to_numpy().ravel(order="F"), name="game_attent"
    )

    print("Game skill VS AI skill:")
    wilcox_df[context + "_skill"] = calc_ttest(
        pd.concat([game_intent_df, game_releva_df]),
        pd.concat([ai_intent_df, ai_releva_df]),
    )

    print("Game attent VS AI attent:")
    wilcox_df[context + "_attention"] = calc_ttest(game_attent_df, ai_attent_df)

    ai_drawn_df = ai_df.filter(regex="drawn", axis=1)
    game_drawn_df = game_df.filter(regex="drawn", axis=1)
    ai_drawn_df = pd.Series(ai_drawn_df.to_numpy().ravel(order="F"), name="ai_drawn")
    game_drawn_df = pd.Series(
        game_drawn_df.to_numpy().ravel(order="F"), name="game_drawn"
    )

    print("Game drawn VS AI drawn:")
    wilcox_df[context + "_drawn"] = calc_ttest(game_drawn_df, ai_drawn_df)

    print("Game FA VS AI FA:")
    wilcox_df[context + "_FA"] = calc_ttest(
        pd.concat([game_attent_df, game_drawn_df]),
        pd.concat([ai_attent_df, ai_drawn_df]),
    )
    # calc_ttest(game_df, ai_df)
    # df = pd.concat([game_df, ai_df], axis=1)
    df = pd.concat(
        [
            ai_releva_df,
            game_releva_df,
            ai_intent_df,
            game_intent_df,
            ai_attent_df,
            game_attent_df,
            ai_drawn_df,
            game_drawn_df,
        ],
        axis=1,
    )
# df = og_df.fillna(df.median())
# og_df.dropna(inplace=True)
# df = og_df.astype(int)
# print(wilcox_df.head(6))
wilcox_df.to_csv("dialog_stats_alll.csv")
# exit()

# df = df.filter(regex=r"ai_", axis=1)
# df = preprocessing.scale(X=df)
# df = pd.DataFrame(data=df)
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity

# df = og_df
df = df.filter(regex=r"ai_|game_", axis=1)
# df.dropna(inplace=True)
df = df.fillna(df.median())

chi_square_value, p_value = calculate_bartlett_sphericity(df)
print("chi:", chi_square_value, "p:", p_value)
from factor_analyzer.factor_analyzer import calculate_kmo

kmo_all, kmo_model = calculate_kmo(df)
print("kmo", kmo_model)

fa = FactorAnalyzer(n_factors=2, rotation="varimax")
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
# plt.show()

sns.heatmap(
    data=df.corr(),
    vmin=-1,
    vmax=1,
    center=0,
    # cmap=sns.diverging_palette(20, 220, n=470),
    square=True,
)
##plt.show()
print(df.info())
print(pd.DataFrame.from_records(loadings))
# print(fa.corr_)
# print(fa.rotation_matrix_)
print(fa.get_factor_variance())
# rint(fa.transform(X=df))
