import pandas as pd
import ast
import numpy as np
import re
import string
def clean_source(source):
    #source = [str(s)[:-1] if s[-1] == '"' else s for s in source]
    source = [re.sub("\[[^>]+\]", "", str(s)) for s in source]
    source = [re.sub("\{[^>]+\}", "", s) for s in source]
    source = [re.sub("\([^>]+\)", "", s) for s in source]

    source = [re.sub("\{\}", "", s) for s in source]
    source = [re.sub("\[\]", "", s) if s[:-1] != "]" else s for s in source ]
    source = [str(s).replace("\r", "") for s in source]
    source = [str(s).replace("&apos;", "'") for s in source]
    source = [str(s).replace("\n", "") for s in source]
    source = [str(s).replace("&quot;", '"') for s in source]
    source = [str(s).replace("quot;", '"') for s in source]
    source = [str(s).replace("lt;StartAction", "") for s in source]
    source = [str(s).replace("PERSUASION : ", "") for s in source]
    source = [str(s).replace("&lt;", "") for s in source]
    source = [str(s).replace("&gt;", "") for s in source]
    source = [str(s).replace("</string", "") for s in source]
    source = [str(s).replace("StartCheck/Start", "") for s in source]
    source = [str(s).replace("StartAction/Start", "") for s in source]
    source = [str(s).replace("StartCheckEingebung/Start", "") for s in source]
    source = [str(s).replace("StartActionIntuizione/Start", "") for s in source]
    source = [str(s).replace("StartCheckEingebung/Start", "") for s in source]
    source = [str(s).replace("StartCheckEingebung/Start", "") for s in source]
    source = [str(s).replace("StartCheckEingebung/Start", "") for s in source]
    source = [str(s).replace(">", "") for s in source]
    source = [str(s).replace("<", "") for s in source]

    source = [s.rstrip() for s in source]
    source = [x for x in source if x]
    source = [s.lstrip(string.punctuation + string.whitespace) for s in source]
    source = [str(s).replace(">", "") for s in source]

    if len(source) > 0:
        return source[0]
    if len(source) == 0:
        return None
def rm_unnecessary_columns(df):
    col_list = [
        "game",
        "filename",
        "id",
        "sid",
        "replies",
        # "entries",
        "persuasion",
        "en",
    ]
    return df.drop(columns=[col for col in df if col not in col_list])


def unify_persuasion(df):
    df.persuasion = df.persuasion.fillna(False)
    # df.persuasion = df.persuasion.apply(ast.literal_eval)
    for i, row in df.iterrows():
        if type(row["persuasion"]) != bool:
        
            if "[" not in row["persuasion"]:
                #df.at[i, "persuasion"] = str(row["persuasion"])
                df.at[i,'persuasion'] = str(df.at[i,'persuasion'])
            elif row["persuasion"] == "[]":
                df.at[i, "persuasion"] = False
            else:
                #df.at[i, "persuasion"] = ast.literal_eval(row["persuasion"])
                row = row['persuasion'].strip('][').split(', ')
                row = row[0]
                row = re.sub("'", "", row)
                df.at[i, "persuasion"] = row

            try:
                test = row['persuasion'].__contains__("Persuade")
                test2  = row['persuasion'].__contains__("/")
                test3 =  row['persuasion'].__contains__("<StartAction>")
                test5 =  row['persuasion'].__contains__("_Handed")
                test6 =  row['persuasion'].__contains__("Spell_")
                if test and test2:
                        df.at[i, "persuasion"] = "Persuade"
                if test3:
                        df.at[i, "persuasion"] = "Persuade"
                if row['persuasion'][0].isdigit():
                        df.at[i, "persuasion"] = False
                if test5:
                        df.at[i, "persuasion"] = "Swords"
                if test6:
                        df.at[i, "persuasion"] = "Magic"


                    
                        
                    
            except TypeError:
                pass
                

    return df


kotor_nwn_df = pd.read_csv("data\persuasion_dialogue_dataset.csv")
kotor_nwn_df = kotor_nwn_df.rename(columns={"is_persuasion": "persuasion"})
poe2_df = pd.read_csv("data\poe2_dataset.csv", sep="\t")
poe1_df = pd.read_csv("data\poe1_dataset.csv", sep="\t")
tyranny_df = pd.read_csv("data\\tyranny_dataset.csv", sep="\t")

df_list = [kotor_nwn_df, poe2_df, poe1_df, tyranny_df]
df_list = [rm_unnecessary_columns(df) for df in df_list]

df = pd.concat(df_list, ignore_index=True)
df = unify_persuasion(df)
for i, row in df.iterrows():
    df.at[i, 'en'] = clean_source([row['en']])
print(df["persuasion"].value_counts())
df = df.dropna(subset=["en"])

df.to_csv('persuasion_dataset.csv', encoding='utf-8', sep="\t")
#319584
#