
import jsonlines
import pandas as pd


def processjsonl(jsonl_path):

    dataset = {"date_publish": [], "title": [], "source_domain": [], "maintext": [], "split": [], "id": []}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in jsonlines.Reader(f):
            dataset["date_publish"].append(line["date_publish"])
            dataset["title"].append(line["title"])
            dataset["source_domain"].append(line["source_domain"])
            dataset["maintext"].append(line["maintext"])
            dataset["split"].append(line["split"])
            dataset["id"].append(line["id"])

    dataset = pd.DataFrame(dataset)

    return dataset

def processjsonltest(jsonl_path):

    dataset = {"date_publish": [], "source_domain": [], "maintext": [], "split": [], "id": []}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in jsonlines.Reader(f):
            dataset["date_publish"].append(line["date_publish"])
            dataset["source_domain"].append(line["source_domain"])
            dataset["maintext"].append(line["maintext"])
            dataset["split"].append(line["split"])
            dataset["id"].append(line["id"])

    dataset = pd.DataFrame(dataset)

    return dataset

