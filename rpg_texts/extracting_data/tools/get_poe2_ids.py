import re
import pandas as pd
import os
import xmltodict, json, xmljson
from lxml.etree import fromstring, tostring
import lxml
import numpy as np
from pprint import pprint


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


def get_node_ids(elem, file):
    # pprint(elem)
    node_id = elem["NodeID"]
    # pprint(elem)
    # print("NODE ID:",node_id)
    from_idx = []
    to_idx = []
    if len(elem["Links"]) > 0:
        for link in elem["Links"]:
            from_idx.append(link["FromNodeID"])
            to_idx.append(link["ToNodeID"])
    if len(elem["Links"]) == 0:
        from_idx.append("no_links")
        to_idx.append("no_links")

    skill_req = []
    skill_diff = []
    for lst in elem["Conditionals"]["Components"]:
        if "Data" in lst.keys():
            # print(lst["Data"])
            if "IsAttributeScoreValue" in lst["Data"]["FullName"]:
                print(lst["Data"]["Parameters"])
                skill_req.append(lst["Data"]["Parameters"][1])
                skill_diff.append(
                    (str(lst["Data"]["Parameters"][2] + lst["Data"]["Parameters"][3]))
                )
            else:
                # skill_req.append("")
                # skill_diff.append("")
                pass

    row = {
        "game": "poe2",
        "filename": file,
        "id": node_id,
        "sid": "",
        "speaker": "",
        "replies": to_idx,
        "entries": from_idx,
        "persuasion": skill_req,
        "difficulty": skill_diff,
    }
    df = pd.DataFrame.from_dict(row, orient="index")
    df = df.transpose()
    if len(skill_req) != 0:
        print(df)
    return df


# dialogue_files = getListOfFiles("extracting_data\eternity_engine\poe2\exported")
attribute_files = getListOfFiles("extracting_data\eternity_engine\poe2\\attr")
# attribute_files = ['extracting_data/eternity_engine/Tyranny/conversations/global/slides_act01_intro.conversation']

df = pd.DataFrame(
    columns=[
        "game",
        "filename",
        "id",
        "sid",
        "speaker",
        "replies",
        "entries",
        "persuasion",
        "difficulty",
    ]
)
# attribute_files = [
#    "extracting_data\\eternity_engine\\poe2\\attr\\lax3_exported\\design\\conversations\\lax03_forgotten_sanctum\\lax03_03_cv_book_pedestal_archive_01.conversationbundle"
# ]
for file in attribute_files:
    # print(file)
    with open(file, "r", encoding="utf-8") as f:
        # o = xmltodict.parse(f)
        # lines = json.dumps(o)
        try:
            f = f.read()
            try:
                lines = json.loads(f)
                if ".conversationbundle" in file:
                    for elem in lines["Conversations"][0]["Nodes"]:
                        # if len(elem["Conditionals"]["Components"]) > 0:
                        # update IDs, reply IDs, and skillchecks
                        df = pd.concat(
                            [get_node_ids(elem, file), df], ignore_index=True
                        )
                        # print(df['persuasion'].value_counts())
                if ".chatterbundle" in file:
                    for elem in lines["ChatterFiles"][0]["Nodes"]:
                        # if len(elem["Conditionals"]["Components"]) > 0:
                        df = pd.concat(
                            [get_node_ids(elem, file), df], ignore_index=True
                        )
            except json.decoder.JSONDecodeError:
                print("json error with", file)
        except UnicodeDecodeError:
            print("Error reading file", file)
filename = "data/only_ids/poe2_dataset_only_ids.csv"
df.to_csv(filename, encoding="utf-8", sep="\t")
