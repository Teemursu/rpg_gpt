import pandas as pd

df = pd.read_csv('data\persuasion_dialogue_dataset.csv')
print(df['persuasion'].value_counts())