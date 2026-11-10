import pandas as pd
import os


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


dfs_to_concat = getListOfFiles("data\data\\to_concat_with_speakers")

df = pd.DataFrame()
for df_to_concat in dfs_to_concat:
    df_to_concat = pd.read_csv(df_to_concat, sep="\t")
    df = pd.concat([df, df_to_concat], ignore_index=True)
try:
    df.drop("Unnamed: 0.1", axis=1, inplace=True)
except KeyError:
    pass
try:
    df.drop("Unnamed: 0", axis=1, inplace=True)
except KeyError:
    pass

df.to_csv("NLG_clean_with_speakers.csv", sep="\t")
