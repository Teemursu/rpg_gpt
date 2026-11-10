import pandas as pd

# Convert JSON File to CSV File
# pandas read JSON File
df = pd.read_json("D:/Code/Repositories/rpg_bot_mahti/data/NLG_RPG.jsonl", lines=True)
print(df.head())
df["Knowledge"] = ""
df.rename(columns={"prompt": "Context", "completion": "Response"}, inplace=True)
df.to_csv("NLG_GODEL.csv")
