
import json
import os
import pandas as pd
import random
from preproc_config import ORIGINAL_DATA_PATH


def generate_label(path, save_path):
    ids = []
    gender = []

    for root, folders, files in os.walk(path):
        if len(folders) == 0:
            json_file_name = [file for file in files if "fq.json" in file][0]
            if json_file_name is not None:
                json_path = os.path.join(root, json_file_name)
                with open(json_path, encoding="utf-8") as f:
                    content = f.readlines()
                target = ["\\a", "\\b", "\\e", "\\f", "\\n", "\\v", "\\t", "\\r", "\\w", "\\x", "\\0"]
                replaced = ["/a", "/b", "/e", "/f", "/n", "/v", "/t", "/r", "/w", "/x", "/0"]
                content = content[0]
                if content[-1] == ",":
                    content = content[:-1] + "]"
                if not content[-1] == "]":
                    content += "]"
                if not content[0] == "[":
                    content = "[" + content
                for i in range(len(target)):
                    if target[i] in content:
                        content = content.replace(target[i], replaced[i])
                json_dict = json.loads(content)
                ids.append(json_dict[0]["uniqueStudentId"])
                gender.append(json_dict[0]["gender"])
    data = {
        "id": ids,
        "gender": gender
    }

    df = pd.DataFrame(data)
    df.to_csv(save_path, sep = ",")

    return ids, gender

def choose_sample(samples, count):
    index = [i for i in range(len(samples))]
    random.shuffle(index)
    return samples[:count]

if __name__ == "__main__":

    healthy_id, healthy_gender = generate_label(ORIGINAL_DATA_PATH["healthy"], "healthy_samples.csv")
    unhealthy_id, unhealthy_gender = generate_label(ORIGINAL_DATA_PATH["unhealthy"], "unhealthy_samples.csv")

    # print(type(healthy_gender[0]))
    healthy_0 = [healthy_id[i] for i in range(len(healthy_id)) if healthy_gender[i] == 0]
    healthy_1 = [healthy_id[i] for i in range(len(healthy_id)) if healthy_gender[i] == 1]
    unhealthy_0 = [unhealthy_id[i] for i in range(len(unhealthy_id)) if unhealthy_gender[i] == 0]
    unhealthy_1 = [unhealthy_id[i] for i in range(len(unhealthy_id)) if unhealthy_gender[i] == 1]
    healthy_count_0 = len(healthy_0)
    healthy_count_1 = len(healthy_1)
    unhealthy_count_0 = len(unhealthy_0)
    unhealthy_count_1 = len(unhealthy_1)
    print("healthy 0:%d, healthy 1:%d, unhealthy 0:%d, unhealthy 1:%d"%(healthy_count_0, healthy_count_1, unhealthy_count_0, unhealthy_count_1))

    count = min([healthy_count_0, healthy_count_1, unhealthy_count_0, unhealthy_count_1])
    healthy_0 = choose_sample(healthy_0, count)
    healthy_1 = choose_sample(healthy_1, count)
    unhealthy_0 = choose_sample(unhealthy_0, count)
    unhealthy_1 = choose_sample(unhealthy_1, count)

    data = {
        "healthy0": healthy_0,
        "healthy1": healthy_1,
        "unhealthy0": unhealthy_0,
        "unhealthy1": unhealthy_1
    }
    df = pd.DataFrame(data)
    df.to_csv("chosen_samples.csv", sep = ",")

    """
    以下涉及对文本数据的再次筛选===========================================================
    """
    # 等比例数据集
    df = pd.read_csv("../data_preprocessed/text_healthy.csv", sep = ",")
    healthy_data = {"id": [],
                "ans0":[],
                "ans1":[],
                "ans2":[],
                "ans3":[],
                "ans4":[]
                }
    for i in range(len(df["ans0"])):
        if df["unique_id"][i] in healthy_0 or df["unique_id"][i] in healthy_1:
            healthy_data["id"].append(df["unique_id"][i])
            healthy_data["ans0"].append(df["ans0"][i])
            healthy_data["ans1"].append(df["ans1"][i])
            healthy_data["ans2"].append(df["ans2"][i])
            healthy_data["ans3"].append(df["ans3"][i])
            healthy_data["ans4"].append(df["ans4"][i])

    healthy_df = pd.DataFrame(healthy_data)
    healthy_df.to_csv("../data_preprocessed/text_healthy_chosen.csv", sep = ",")

    df = pd.read_csv("../data_preprocessed/text_unhealthy.csv", sep=",")
    unhealthy_data = {"id": [],
                    "ans0": [],
                    "ans1": [],
                    "ans2": [],
                    "ans3": [],
                    "ans4": []
                    }
    for i in range(len(df["ans0"])):
        if df["unique_id"][i] in unhealthy_0 or df["unique_id"][i] in unhealthy_1:
            unhealthy_data["id"].append(df["unique_id"][i])
            unhealthy_data["ans0"].append(df["ans0"][i])
            unhealthy_data["ans1"].append(df["ans1"][i])
            unhealthy_data["ans2"].append(df["ans2"][i])
            unhealthy_data["ans3"].append(df["ans3"][i])
            unhealthy_data["ans4"].append(df["ans4"][i])

    unhealthy_df = pd.DataFrame(unhealthy_data)
    unhealthy_df.to_csv("../data_preprocessed/text_unhealthy_chosen.csv", sep=",")

