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


games = ["poe2"]  # "poe1", "poe2", "tyranny"]
langs = ["en"]  # "de", "es", "fr", "it", "pl", "ru", "pt", "zh"]

for game in games:
    with open("data/only_ids/" + game + "_dataset_only_ids.csv", "r") as f:
        df = pd.read_csv(f, sep="\t")
        df["text"] = ""
        files = df.filename.unique()
        # files = [
        #    "extracting_data/eternity_engine/poe2/attr/laxi_exported/design/conversations/laxi/laxi_00_si_god.conversationbundle"
        # ]
        if game == "tyranny":
            files = getListOfFiles(
                "extracting_data\eternity_engine\\tyranny\extrastrings"
            )
    for lang in langs:
        df[lang] = ""
        for filename in files:
            if game != "tyranny":
                orig_filename = filename
                print(orig_filename)
                filename = re.sub("\\\\", "/", filename)
                filename = re.sub(
                    r"/design/",
                    r"/localized/en/text/",
                    filename,
                )
                filename = re.sub(
                    r"/attr/",
                    r"/str/",
                    filename,
                )
                if game == "tyranny" or game == "poe1":
                    filename = re.sub(".conversation", ".stringtable", filename)
                if game == "poe2":
                    filename = re.sub(".chatterbundle", ".stringtable", filename)
                    filename = re.sub(".conversationbundle", ".stringtable", filename)
                print(filename)

            if game == "tyranny":
                orig_filename = re.sub(
                    r"extrastrings\\exported\d\\localized\\en\\",
                    "",
                    filename,
                )
                orig_filename = re.sub(".stringtable", r".conversation", orig_filename)
            try:
                with open(filename, "rb") as f:
                    try:
                        lines = xmltodict.parse(f, dict_constructor=dict)
                    except:
                        print("error reading xml file", filename)
                        break
                    # print(filename, "found!")
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

    df.to_csv("data/" + game + "_dataset.csv", sep="\t")
