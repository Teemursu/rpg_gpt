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

og_df = pd.read_csv("evaluation/survey_data861_anonymised.csv", sep=";")

df = og_df.filter(regex=r"game_|ai_")
print(df.info)
print(df.head(1))
