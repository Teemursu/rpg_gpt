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


def get_node_ids(elem, file, game):
    try:
        node_id = elem["NodeID"]
    except TypeError:
        # print(elem)
        return None
    except KeyError:
        print(elem)
    try:
        asd = len(elem["Links"]["FlowChartLink"])
    except TypeError:
        return None

    try:
        speaker = elem["SpeakerGuid"]
    except KeyError:
        speaker = ""
    # print("NODE ID:",node_id)
    from_idx = []
    to_idx = []
    if type(elem["Links"]["FlowChartLink"]) == dict:
        from_idx.append(elem["Links"]["FlowChartLink"]["FromNodeID"])
        to_idx.append(elem["Links"]["FlowChartLink"]["ToNodeID"])

    if type(elem["Links"]["FlowChartLink"]) == list:
        for lst in elem["Links"]["FlowChartLink"]:
            from_idx.append(lst["FromNodeID"])
            to_idx.append(lst["ToNodeID"])
    else:
        from_idx.append(elem["Links"]["FlowChartLink"]["FromNodeID"])
        to_idx.append(elem["Links"]["FlowChartLink"]["ToNodeID"])

    skill_req = []
    skill_diff = []
    if elem["Conditionals"]["Components"] != None:

        if type(elem["Conditionals"]["Components"]["ExpressionComponent"]) == dict:
            if (
                "Data"
                in elem["Conditionals"]["Components"]["ExpressionComponent"].keys()
            ):
                if (
                    "Boolean IsPlayerScaledSkillValue"
                    in elem["Conditionals"]["Components"]["ExpressionComponent"][
                        "Data"
                    ]["FullName"]
                ):
                    skill_req = elem["Conditionals"]["Components"][
                        "ExpressionComponent"
                    ]["Data"]["Parameters"]["string"]

        if type(elem["Conditionals"]["Components"]["ExpressionComponent"]) == list:
            for lst in elem["Conditionals"]["Components"]["ExpressionComponent"]:

                if "Data" in lst.keys():

                    if lst["Data"]["Parameters"] != None:
                        # print(lst)
                        # print(lst['Data'])
                        if "SkillValue" in lst["Data"]["FullName"]:
                            skill_req = lst["Data"]["Parameters"]["string"]
                else:
                    if lst["Components"] != None:

                        for sublist in lst["Components"]["ExpressionComponent"]:
                            if type(sublist) == dict:
                                if "Data" in sublist.keys():
                                    if "SkillValue" in sublist["Data"]["FullName"]:
                                        skill_req = sublist["Data"]["Parameters"][
                                            "string"
                                        ]

                                else:
                                    if sublist["Components"] != None:

                                        for subsublist in sublist["Components"][
                                            "ExpressionComponent"
                                        ]:
                                            if type(subsublist) == dict:
                                                if "Data" in subsublist.keys():
                                                    if (
                                                        "SkillValue"
                                                        in subsublist["Data"][
                                                            "FullName"
                                                        ]
                                                    ):
                                                        skill_req = subsublist["Data"][
                                                            "Parameters"
                                                        ]["string"]

    if len(skill_req) != 0:
        if game == "tyranny":
            persuasion = skill_req[1]
            difficulty = skill_req[2]
        if game == "poe1":
            persuasion = skill_req[0]
            difficulty = skill_req[2]
        print(game, persuasion, difficulty)
    else:
        persuasion = ""
        difficulty = ""
    row = pd.DataFrame(
        {
            "game": game,
            "filename": file,
            "id": node_id,
            "sid": "",
            "speaker": speaker,
            "replies": to_idx,
            "entries": from_idx,
            "persuasion": persuasion,
            "difficulty": difficulty,
        }
    )
    return row


games = ["tyranny", "poe1"]
for game in games:
    dialogue_files = getListOfFiles(
        "extracting_data\eternity_engine\{}\exported".format(game)
    )
    attribute_files = getListOfFiles(
        "extracting_data\eternity_engine\{}\conversations".format(game)
    )
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

    for file in attribute_files:
        # print(file)
        with open(file, "rb") as f:
            o = xmltodict.parse(f)
            lines = json.dumps(o)
            lines = json.loads(lines)
            for elem in lines["ConversationData"]["Nodes"]["FlowChartNode"]:

                # update IDs, reply IDs, and skillchecks
                df = pd.concat([get_node_ids(elem, file, game), df], ignore_index=True)
                # print(df['persuasion'].value_counts())
                # print(df)

    filename = "data/only_ids/" + game + "_dataset_only_ids.csv"
    df.to_csv(filename, encoding="utf-8", sep="\t")
