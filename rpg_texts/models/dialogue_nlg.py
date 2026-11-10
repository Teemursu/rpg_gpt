import pandas as pd


def read_squad(path):
    # open JSON file and load intro dictionary
    with open(path, "rb") as f:
        squad_dict = json.load(f)

    # initialize lists for contexts, questions, and answers
    contexts = []
    questions = []
    answers = []
    # iterate through all data in squad data
    for group in squad_dict["data"]:
        for passage in group["paragraphs"]:
            context = passage["context"]
            for qa in passage["qas"]:
                question = qa["question"]
                # check if we need to be extracting from 'answers' or 'plausible_answers'
                if "plausible_answers" in qa.keys():
                    access = "plausible_answers"
                else:
                    access = "answers"
                for answer in qa[access]:
                    # append data to lists
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    # return formatted data lists
    return contexts, questions, answers
