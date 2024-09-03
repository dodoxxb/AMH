"""
This file is used to preprocess the text data.

extract data from json file.
store useful data to csv file.
- csv file:
stu_id, ans0-7, path0-4

统计每个人说的话有多长
question is： 每个问题长度限定还是只有总长度限定？
我觉得每个问题限定比较好

data path:  ../data/data_ok/healthy(or unhealthy)
            --emotionData2
              |--unique_id1
                 |--json_file(including the text data, the name of the json file inlcudes "fq.json")
              |--unique_id2

for every json file:
    each participant were asked 9 questions.
    there are general 2 kinds of questions: Q&A, multiple choice.
    the multiple choice questions's "options" key is not ""(empty string).

    for each question:
    - age: age of the student
    (important)- answer: student's answer to this question
    - classGrade: eg初一1班
    - fileUrl: 没用
    - gender: 性别
    - id:
    (important)- localPath: wav文件存储的绝对路径，只有文件名有用
    - missionId:
    - name: 随意姓名
    - officeId:
    - options: 选择题的选项写在这，简答题为""
    - question: 问题内容
    - studentId: 学号
    (important)- uniqueStudentId: unique学号，和文件夹名字相同
"""

# imports
import os
import json
import pandas as pd
import difflib
import jieba
from preproc_config import ORIGINAL_DATA_PATH, TEXT_CSV_PATH

questions={"简单介绍下自己的兴趣和爱好。": 0,
               "你最喜欢的偶像是谁？为什么呢？": 1,
               "你最喜欢的偶像是谁？": 1,
               "你最崇拜谁？为什么呢？": 2,
               "你最崇拜谁？": 2,
               "请分享一件最近最开心的事情。": 3,
               "请分享一件最近最困扰你的或者让你最不开心的事情。": 4,
               "我今年已经上初中了，但是周围的同学我都不熟悉，没人跟我交朋友。如果你是小A，你会怎么做？": 5,
               " 我今年已经上初中了，但是周围的同学我都不熟悉，没人跟我交朋友。如果你是小A，你会怎么做？": 5,
               "为什么我都已经这么努力了，却还是最后一名，我应该真的是太笨了，不适合学习。如果你是小A，你会怎么做": 6,
               "这是我的家，他们是我的父母，我每天都在这样的争吵中度过，我如果你是小A，你会怎么做？": 7,
               "这是我的家，他们是我的父母，我每天都在这样的争吵中度过，如果你是小A，你会怎么做？": 7,
               "这是我的家，他们是我的父母，我今年12岁了，每天都在这样的争吵中红度过，我如果你是小A，你会怎么做？": 7,
               "游戏图片": 8
               }
def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

def extract_text(path, save_csv_path):
    unique_id = []
    ans = [[], [], [], [], [], [], [], []]
    paths = [[], [], [], [], []]

    for root, _, files in os.walk(path):
        file = ""
        for _ in files:
            if "fq.json" in _:
                file = _
                break
        if not file == "":
            json_path = os.path.join(root, file)
            print(json_path)
            with open(json_path, encoding="utf-8") as f:
                content = f.readlines()
            """
            接下来是丑陋的处理
            读进来的字符串有各种问题（没有问题我就直接读json文件了好伐
            'if .. else ..', the most powerful AI in the world.
            """
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
            # now content is str.
            # why replace? because \e \0 \w exists in the file path (localPath or fileUrl) but 会被识别成转义字符
            # "\" is used in win path, there would be no such problem in linux/unix systems (since they use "/")
            # 如果把所有的\都替换，options里面的小字典无法被识别
            # 出现abefnvtro0wx需要替换（可能还有其它的）
            json_dict = json.loads(content)  # json_dict is a list, containing 9 dicts
            # print("json file loaded: ",json_path)
            # how to deal with each dict?
            # csv文件
            # 学生uniqueid, ans0-4, choice6-8(0 or 1, no answer = 0.5), (wav_paths)
            unique_id.append(json_dict[0]["uniqueStudentId"])

            hash_table = [0, 0, 0, 0, 0, 0, 0, 0]  # some of the question may not be recorded

            for item in json_dict:
                question = item["question"]
                question_id = questions[question]  # 0-4问答，5-7选择

                if (not question_id == 8) and (hash_table[question_id] == 0):
                    hash_table[question_id] = 1
                    if question_id < 5:
                        paths[question_id].append(item["localPath"])
                        ans[question_id].append(item["answer"])
                    elif question_id < 8:
                        answer = item["answer"]
                        option = json.loads(item["options"])
                        if answer == "":
                            ans[question_id].append(0.5)
                        else:
                            ans0 = option[0]["content"]
                            ans1 = option[1]["content"]
                            score0 = string_similar(ans0, answer)
                            score1 = string_similar(ans1, answer)
                            if score0 > score1:
                                ans[question_id].append(0)
                            else:
                                ans[question_id].append(1)
            for i in range(len(hash_table)):
                if hash_table[i] == 0:
                    # the question i is not answered
                    if i < 5:
                        ans[i].append("")
                        paths[i].append("")
                    elif i < 8:
                        ans[i].append(0.5)

    data = {
        "unique_id": unique_id,
        "ans0": ans[0],
        "ans1": ans[1],
        "ans2": ans[2],
        "ans3": ans[3],
        "ans4": ans[4],
        "ans5": ans[5],
        "ans6": ans[6],
        "ans7": ans[7],
        "path0": paths[0],
        "path1": paths[1],
        "path2": paths[2],
        "path3": paths[3],
        "path4": paths[4]
    }
    df = pd.DataFrame(data,
                      columns=["unique_id", "ans0", "ans1", "ans2", "ans3", "ans4", "ans5", "ans6", "ans7", "path0",
                               "path1", "path2", "path3", "path4"])
    df.to_csv(save_csv_path, sep=",")

def count_word(path, save_csv_path):
    df = pd.read_csv(path, sep = ",")
    ids = df["unique_id"]
    ans0 = df["ans0"]
    ans1 = df["ans1"]
    ans2 = df["ans2"]
    ans3 = df["ans3"]
    ans4 = df["ans4"]
    anss = [ans0, ans1, ans2, ans3, ans4]

    word_nums = [[], [], [], [], []]
    for i in range(len(ans0)):
        for j in range(5):
            if isinstance(anss[j][i], str):
                word_nums[j].append(len(jieba.lcut(anss[j][i].strip())))
                print(jieba.lcut(anss[j][i].strip()))
            else:
                word_nums[j].append(0)
    data = {
        "unique_id": ids,
        "ans0": word_nums[0],
        "ans1": word_nums[1],
        "ans2": word_nums[2],
        "ans3": word_nums[3],
        "ans4": word_nums[4]
    }
    df = pd.DataFrame(data,
                      columns=["unique_id", "ans0", "ans1", "ans2", "ans3", "ans4"])
    df.to_csv(save_csv_path, sep=",")

if __name__=="__main__":
    extract_text(ORIGINAL_DATA_PATH["healthy"], TEXT_CSV_PATH["healthy"])
    extract_text(ORIGINAL_DATA_PATH["unhealthy"], TEXT_CSV_PATH["unhealthy"])