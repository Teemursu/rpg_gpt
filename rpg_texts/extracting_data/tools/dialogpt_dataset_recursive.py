import ast
from re import sub
import pandas as pd
import sys, threading
import os
import glob
import numpy as np


def get_replies_poe2_eng(reply_id, conversation, recur_break_list):
    # recur_break_list = []
    if len(recur_break_list) > 2:
        if recur_break_list[-1] in recur_break_list[:-1]:
            return [[]]
    row = conversation[conversation["id"] == str(reply_id)]
    # #print(row["id"].iloc[0], type(row["id"].iloc[0]))
    row = row.drop_duplicates(keep="first")
    if len(row) == 0:
        return [[]]
    # replylist = list(set(row["replies"].tolist()))
    # #print(row["replies"].iloc[0])
    replylist = ast.literal_eval(row["replies"].iloc[0])
    list_of_texts = []
    # replylist.append("no_links")

    for reply in replylist:
        reply = str(reply)
        speaker = row["speaker"].iloc[0]
        if type(speaker) == float:
            speaker = "None"
        recur_break_list.append(reply)
        if reply == "no_links":
            # Change text to include special <Persuade> token if exists
            if "False" not in row["persuasion"].iloc[0]:
                # print(row["persuasion"].iloc[0], type(row["persuasion"].iloc[0]))

                text = "<|{}|>".format(
                    row["persuasion"].iloc[0]
                ).lower() + " {}".format(row["en"].iloc[0])
                # print(row["persuasion"].iloc[0], type(row["persuasion"].iloc[0]))

            if row["persuasion"].iloc[0] == "False":
                text = row["en"].iloc[0]

            # #print(text, "NO MORE RECURSION CALL")
            if type(text) == float:
                return [[]]
            if type(text) == str:
                return [[(text, speaker)]]
        for item in get_replies_poe2_eng(
            reply,
            conversation,
            recur_break_list
            # list_of_texts,
        ):

            # Change text to include special <Persuade> token if exists
            if "False" not in row["persuasion"].iloc[0]:
                text = "<|{}|>".format(
                    row["persuasion"].iloc[0]
                ).lower() + " {}".format(row["en"].iloc[0])
                # #print(text)
                ##print(row["persuasion"].iloc[0], type(row["persuasion"].iloc[0]))

            if row["persuasion"].iloc[0] == "False":
                text = row["en"].iloc[0]
            if type(text) == float:
                list_of_texts.append([] + item)
            if type(text) == str:
                list_of_texts.append([(text, speaker)] + item)
    return list_of_texts


def get_replies_poe1tyranny(reply_id, conversation, recur_break_list):
    # recur_break_list = []
    if len(recur_break_list) > 2:
        if recur_break_list[-1] in recur_break_list[:-1]:
            return [[]]
    row = conversation[conversation["id"] == int(reply_id)]
    row = row.drop_duplicates(keep="first")

    if len(row) == 0:
        return [[]]
    # replylist = list(set(row["replies"].tolist()))
    replylist = [row["replies"].iloc[0]]

    # replylist = ast.literal_eval(row["replies"].iloc[0])
    list_of_texts = []
    # replylist.append("no_links")
    # #print(replylist, type(replylist))

    for reply in replylist:
        reply = str(reply)
        speaker = row["speaker"].iloc[0]
        if type(speaker) == float:
            speaker = "None"
        # #print(speaker)
        recur_break_list.append(reply)
        # #print(reply, type(reply), replylist, recur_break_list)
        if reply == "no_links":
            # Change text to include special <Persuade> token if exists
            if "False" not in row["persuasion"].iloc[0]:
                text = "<|{}|>".format(
                    row["persuasion"].iloc[0]
                ).lower() + " {}".format(row["en"].iloc[0])
                # #print(text)

            if row["persuasion"].iloc[0] == "False":
                text = row["en"].iloc[0]
            # #print(text, "NO MORE RECURSION CALL")
            return [[(text, speaker)]]
        for item in get_replies_poe1tyranny(
            reply,
            conversation,
            recur_break_list
            # list_of_texts,
        ):

            # Change text to include special <Persuade> token if exists
            if "False" not in row["persuasion"].iloc[0]:
                text = "<|{}|>".format(
                    row["persuasion"].iloc[0]
                ).lower() + " {}".format(row["en"].iloc[0])
                # print(text)
                # print(row["persuasion"].iloc[0], type(row["persuasion"].iloc[0]))

            if row["persuasion"].iloc[0] == "False":
                text = row["en"].iloc[0]
            # #print(text)
            # #print("RECUR AGAIN")
            list_of_texts.append([(text, speaker)] + item)
    return list_of_texts


def get_replies_infinity_eng(reply_id, conversation, recur_break_list):
    if len(recur_break_list) > 2:
        if recur_break_list[-1] in recur_break_list[:-1]:
            return [[]]
    row = conversation[conversation["sid"] == np.float64(reply_id)]
    # #print(row)
    # #print("hi")
    # #print(conversation)
    # #print(reply_id, type(reply_id), type(conversation["sid"].iloc[0]))
    # #print(row)
    list_of_texts = []
    try:
        # replylist = [reply_id]
        replylist = ast.literal_eval(row["replies"].iloc[0])
    except IndexError:
        return list_of_texts
    if len(replylist) == 0:
        replylist.append("EoC")
    for reply in replylist:
        reply = str(reply)
        speaker = row["speaker"].iloc[0]
        if type(speaker) == float:
            speaker = "None"
        recur_break_list.append(reply)
        # #print(reply)
        if reply == "EoC":
            # Change text to include special <Persuade> token if exists
            if "False" not in row["persuasion"].iloc[0]:
                text = "<|{}|>".format(
                    row["persuasion"].iloc[0]
                ).lower() + " {}".format(row["en"].iloc[0])
            if row["persuasion"].iloc[0] == "False":
                text = row["en"].iloc[0]
            return [[(text, speaker)]]

        for item in get_replies_infinity_eng(reply, conversation, recur_break_list):
            # Change text to include special <Persuade> token if exists
            if "False" not in row["persuasion"].iloc[0]:
                text = "<|{}|>".format(
                    row["persuasion"].iloc[0]
                ).lower() + " {}".format(row["en"].iloc[0])
            if row["persuasion"].iloc[0] == "False":
                text = row["en"].iloc[0]

            list_of_texts.append([(text, speaker)] + item)
    return list_of_texts


def join_turns_if_same_speaker(lst):
    for item in lst:
        if type(item) != tuple:
            raise TypeError("Each turn in the convo must be a tuple")
    result = []
    prev = None
    curr_str = ""
    player_options = ["b1a8e901-0000-0000-0000-000000000000", "PC"]
    # print("this becomes:", lst)
    for tup in lst:
        # #print(tup)
        if prev is None or tup[1] != prev:
            if curr_str:
                result.append(curr_str)

            if tup[1] in player_options:
                curr_str = "Player: " + str(tup[0])
            elif tup[1] == "None" and prev not in player_options:
                curr_str = "Player: " + str(tup[0])
                # #print("hello")
            else:
                curr_str = "NPC: " + str(tup[0])
        else:
            curr_str += " " + str(tup[0])

        prev = tup[1]
    result.append(curr_str)
    # print("this:", result)
    # print()
    return result


def remove_folder_path_from_filename(filepath):
    return filepath.split("/")[-1].split("\\")[-1]


def get_context_row(convrsation, filename):
    global df
    global context_df
    poe1tyranny = ["tyranny", "poe1"]
    inf_eng_games = ["KotOR 1", "KotOR 2", "NWN 1"]
    conv_df = pd.DataFrame()
    # #print(df.head(5))
    # basename, ext = os.path.splitext(filename)

    for i, row in conversation.iterrows():
        if type(row["en"]) == float:
            row["en"] == "<emptystr>"
        if True:
            # replylist = ast.literal_eval(row["replies"])
            # for reply_id in replylist:
            if row["game"] in inf_eng_games:
                context_list = get_replies_infinity_eng(row["sid"], conversation, [])
            if row["game"] == "poe2":
                context_list = get_replies_poe2_eng(row["id"], conversation, [])
                # #print("resulting context_list", context_list)
            if row["game"] in poe1tyranny:
                context_list = get_replies_poe1tyranny(row["id"], conversation, [])
            for convo in context_list:
                # #print("convo before:\n", convo)
                convo = join_turns_if_same_speaker(convo)
                # #print("convo after:\n", convo)
                # #print()
                if len(convo) > 1:
                    for sent in convo:
                        # #print("-", sent)
                        pass
                    # #print()
                    convo.reverse()
                    temp_df = pd.DataFrame(
                        [convo],
                        columns=["response"]
                        + ["context/{}".format(i) for i in range(len(convo) - 1)],
                    )
                    temp_df["game"] = row["game"]
                    # #print(temp_df)
                    # temp_df.loc[:, "filename"] = filename
                    # context_df = pd.concat([temp_df, context_df], ignore_index=True)
                    conv_df = pd.concat([temp_df, conv_df], ignore_index=True)
                    # context_df = context_df.drop_duplicates()
        # context_df = context_df.dropna()
        # #print(context_df)
        # #print(context_df.shape)

        # df_to_save = context_df[context_df["filename"] == old_filename]
        # df_to_save = df_to_save.drop("filename", 1)
        # df_to_save = df_to_save.drop("filename", 1)
    # #print("saving df for file", filename)
    # #print("df is shape", conv_df.shape)
    if conv_df.empty:
        return None
    conv_df = conv_df.drop_duplicates(keep="first")
    # #print(conv_df, filename)

    # #print("df is shape (after dup drop)", conv_df.shape)

    df_to_save = conv_df
    # #print(df_to_save)
    to_save_filename = os.path.basename(filename)
    # #print(old_filename)
    # #print("turns into")
    # #print(old_filename.replace("\\", " ").split()[-1])
    # #print(to_save_filename)
    df_to_save.to_csv(
        "D:\Code\Data\\rpg_texts\data\data\\to_concat_with_speakers\\{}_with_speakers.csv".format(
            to_save_filename
        ),
        encoding="utf-8",
        sep="\t",
    )

    # #print(context_df.keys())
    # #print("replyid changed to", replylist)


# sys.setrecursionlimit(1000000000)
# threading.stack_size(2**27)  # new thread will get stack of such size

df = pd.read_csv(
    "data\\persuasion_dataset_w_quotes_and_squarebrackets_and_more_skills_with_speakers.csv",
    encoding="utf-8",
    sep="\t",
)

# df = df.iloc[2315]
# df = df[df["filename"] == "extracting_data/infinity_engine/nwn/xml/q1dlodar.dlg.xml"]
# #print(df.head(5))
# #print(df.head(5))

# df = pd.read_csv("data\\toydata.csv", sep=";")
# context_df = pd.Dataframe(columns=['response', 'context/0', 'context/1','context/2','context/3','context/4','context/5','context/6'])
context_df = pd.DataFrame()


lst_games = ["poe1", "poe2", "tyranny", "KotOR 1", "KotOR 2", "NWN 1"]

# #print("filenames", filenames)
df = df.query("game in @lst_games")
all_filenames = list(set(df["filename"].tolist()))
unique_filenames = []
for filename in all_filenames:
    basename = remove_folder_path_from_filename(filename)
    # print(basename)
    glob_check = glob.glob(
        "D:\\Code\\Data\\rpg_texts\\data\\data\\to_concat_with_speakers\\{}_with_speakers.csv".format(
            os.path.basename(basename)
        )
    )
    if len(glob_check) > 0:
        # print(basename, glob_check, "exists!")
        # print(os.path.basename(basename), "csv already found, skipping...")
        pass
        # pass
    else:
        # #print(filename, "does not exist, continuing with conv extraction")
        unique_filenames.append(filename)
for i, filename in enumerate(unique_filenames):
    print("Processing files:", i, "/", len(unique_filenames))  # "//", filename)
    conversation = df[df["filename"] == filename]
    get_context_row(conversation, filename)

# context_row = get_context_row(0, df.iloc[0])

# context_path = "D:/Code/Data/rpg_texts/data/data/NLG_dataset_poe1.csv"
# context_df.to_csv(context_path, encoding="utf-8", sep="\t")
