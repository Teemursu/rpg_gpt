import pandas as pd

df = pd.read_csv('data\kotor_dataset.csv')
print(df['speaker'].value_counts())