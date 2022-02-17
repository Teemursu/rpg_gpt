import ast
from re import sub
import pandas as pd
from pprint import pprint
import sys, threading

# sys.setrecursionlimit(1000000000)
# threading.stack_size(2**27)  # new thread will get stack of such size

df = pd.read_csv("data\persuasion_dataset.csv", encoding="utf-8", sep="\t")
# df = df.iloc[2315]
df = df[
    df["filename"] == "extracting_data/infinity_engine/kotor1/xml/dan14_handon.dlg.xml"
]
print(df)

# df = pd.read_csv("data\\toydata.csv", sep=";")
# context_df = pd.Dataframe(columns=['response', 'context/0', 'context/1','context/2','context/3','context/4','context/5','context/6'])
context_df = pd.DataFrame()


def get_replies(reply_id, conversation):
    row = conversation[conversation["sid"] == float(reply_id)]
    list_of_texts = []
    replylist = ast.literal_eval(row["replies"].iloc[0])
    print(row["en"].item(), reply_id)
    if len(replylist) == 0:
        replylist.append("EoC")
    for reply in replylist:
        print(reply, replylist)
        if reply == "EoC":

            # Change text to include special <Persuade> token if exists
            if "no_persuasion" not in row["persuasion"].iloc[0]:
                text = "<{}>".format(row["persuasion"].iloc[0]) + " {}".format(
                    row["en"].iloc[0]
                )
            if row["persuasion"].iloc[0] == "no_persuasion":
                text = row["en"].iloc[0]
            return [[text]]
        try:
            for item in get_replies(reply, conversation):

                # Change text to include special <Persuade> token if exists
                if "no_persuasion" not in row["persuasion"].iloc[0]:
                    text = "<{}>".format(row["persuasion"].iloc[0]) + " {}".format(
                        row["en"].iloc[0]
                    )
                if row["persuasion"].iloc[0] == "no_persuasion":
                    text = row["en"].iloc[0]
                list_of_texts.append([text] + item)
                for lst in list_of_texts:
                    print(len(lst), lst)
        except RecursionError:
            pass

    return list_of_texts


# [a, [b, d], [b, c]]
# [[a, b, d], [a, b, c]]


def get_context_row(i, row):
    global df
    global context_df
    lst_games = ["KotOR 1", "KotOR 2", "NWN 1"]
    filename = row["filename"]
    conversation = df[df["filename"] == filename]

    if row["game"] in lst_games:
        # replylist = ast.literal_eval(row["replies"])
        # for reply_id in replylist:
        context_list = get_replies(row["sid"], conversation)
        for convo in context_list:
            if len(convo) > 1:
                for sent in convo:
                    print("-", sent)
                print()
                convo.reverse()
                temp_df = pd.DataFrame(
                    [convo],
                    columns=["response"]
                    + ["context/{}".format(i) for i in range(len(convo) - 1)],
                )
                context_df = pd.concat([temp_df, context_df], ignore_index=True)
                print(context_df.shape)

                # print(context_df.keys())
            # print("replyid changed to", replylist)


for i, current_row in df.iterrows():
    if type(current_row["en"]) != float:
        print("new row here")
        context_row = get_context_row(i, current_row)


# context_row = get_context_row(0, df.iloc[0])

context_path = "D:/Code/Data/rpg_texts/data/data/NLG_dataset.csv"
context_df.to_csv(context_path, encoding="utf-8", sep="\t")
