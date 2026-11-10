import pandas as pd

df = pd.read_csv(
    "D:/Code/Repositories/rpg_bot/data/nlg_data_10_turns_all_data.csv", sep="\t"
)
print(df.shape)


df = df[df["response"].str.contains(r"([A-Z]){3}") == False]
df = df[df["context/0"].str.contains(r"([A-Z]){3}") == False]
df = df[df["context/1"].str.contains(r"([A-Z]){3}") == False]
df = df[df["context/2"].str.contains(r"([A-Z]){3}") == False]
df = df[df["context/3"].str.contains(r"([A-Z]){3}") == False]
df = df[df["context/4"].str.contains(r"([A-Z]){3}") == False]
df = df[df["context/5"].str.contains(r"([A-Z]){3}") == False]
df = df[df["context/6"].str.contains(r"([A-Z]){3}") == False]
df = df[df["context/7"].str.contains(r"([A-Z]){3}") == False]
df = df[df["context/8"].str.contains(r"([A-Z]){3}") == False]
# df = df[df["response"].str.contains("GOTO COMMAND CONSOLE") == False]
# df = df[df["response"].str.contains("GOTO COMMAND CONSOLE") == False]
# df = df[df["context/0"].str.contains("SYSTEM CONSOLE") == False]
# df = df[df["response"].str.contains("SECURITY CONSOLE") == False]
# df = df[df["context/0"].str.contains("SECURITY CONSOLE") == False]
# df = df[df["response"].str.contains("GOTO COMMAND CONSOLE") == False]
# df = df[df["context/0"].str.contains("SYSTEM CONSOLE") == False]
# df = df[df["response"].str.contains("ROOM ACCESS RESTRICTED") == False]
# df = df[df["context/0"].str.contains("ROOM ACCESS RESTRICTED") == False]
# df = df[df["response"].str.contains(" SYSTEM ") == False]
# df = df[df["context/0"].str.contains(" SYSTEM ") == False]
# df = df[df["response"].str.contains(" ACCESS ") == False]
# df = df[df["context/0"].str.contains(" ACCESS ") == False]


print(df.shape)

df.to_csv(
    "D:/Code/Repositories/rpg_bot/data/nlg_data_10_turns_all_data_without_consoledialogue.csv",
    sep="\t",
)
