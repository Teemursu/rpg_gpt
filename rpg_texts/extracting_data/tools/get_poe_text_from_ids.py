from pyexpat import ExpatError
import re
import pandas as pd
import os
import xmltodict, json, xmljson
from lxml.etree import fromstring, tostring
import lxml
from pprint import pprint
import xml.etree.ElementTree as ET
import numpy as np


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


games = ["poe1", "poe2", "tyranny"]
langs = ["en", "de", "es", "fr", "it", "pl", "ru", "pt", "zh"]

for game in games:
    with open("data/only_ids/" + game + "_dataset_only_ids.csv", "r") as f:
        df = pd.read_csv(f)
        df["text"] = ""
        files = df.filename.unique()
    for lang in langs:
        df[lang] = ""
        for filename in files:
            orig_filename = filename
            filename = re.sub("\\\\", "/", filename)
            filename = re.sub(
                "conversations",
                "exported/localized/{}/text/conversations".format(lang),
                filename,
            )
            if game == "tyranny" or game == "poe1":
                filename = re.sub(r"\.conversation", ".stringtable", filename)
            if game == "poe2":
                filename = re.sub(r"\.conversationbundle", ".stringtable", filename)
            try:
                with open(filename, "rb") as f:
                    try:
                        lines = xmltodict.parse(f, dict_constructor=dict)
                    except:
                        print("error reading xml file", filename)
                        break
                    print(filename, "found!")
                    entrydict = lines["StringTableFile"]["Entries"]

                    if type(entrydict["Entry"]) == dict:
                        entries = [entrydict["Entry"]]
                    if type(entrydict["Entry"]) == list:
                        entries = entrydict["Entry"]

                    for entry in entries:
                        df.loc[
                            (df["id"] == np.int64(entry["ID"]))
                            & (df["filename"] == orig_filename),
                            lang,
                        ] = entry["DefaultText"]

            except FileNotFoundError:
                # print(file, "not found")
                pass

    df.to_csv("data/" + game + "_dataset.csv")
