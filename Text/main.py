# coding=utf-8

import os
import mediapipe as mp
import cv2 as cv
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

def pert_get_feature(text_list, list_len):
    # Download pretrained model from Internet to "\.cache\torch\transformers\"
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-pert-large')
    bert = BertModel.from_pretrained('hfl/chinese-pert-large')

    # text_list = ["中文", "自然语言处理"]
    inputs = tokenizer(text_list, return_tensors="pt", padding=True)  # "pt"表示"pytorch"
    outputs = bert(**inputs)

    # print(outputs)
    print(outputs.pooler_output)
    print(outputs.pooler_output.shape)
    text_ft = outputs.pooler_output.detach().numpy().reshape(list_len, -1)
    print(text_ft.shape)
    return text_ft

def roberta_get_feature(text_list, list_len):
    # Download pretrained model from Internet to "\.cache\torch\transformers\"
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext-large')
    bert = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext-large')

    # text_list = ["中文", "自然语言处理"]
    inputs = tokenizer(text_list, return_tensors="pt", padding=True)  # "pt"表示"pytorch"
    outputs = bert(**inputs)

    # print(outputs)
    print(outputs.pooler_output)
    print(outputs.pooler_output.shape)
    text_ft = outputs.pooler_output.detach().numpy().reshape(list_len, -1)
    print(text_ft.shape)
    return text_ft

def electra_get_feature(text_list, list_len):
    # Download pretrained model from Internet to "\.cache\torch\transformers\"
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-electra-180g-large-discriminator')
    bert = BertModel.from_pretrained('hfl/chinese-electra-180g-large-discriminator')

    # text_list = ["中文", "自然语言处理"]
    inputs = tokenizer(text_list, return_tensors="pt", padding=True)  # "pt"表示"pytorch"
    outputs = bert(**inputs)

    # print(outputs)
    print(outputs.pooler_output)
    print(outputs.pooler_output.shape)
    text_ft = outputs.pooler_output.detach().numpy().reshape(list_len, -1)
    print(text_ft.shape)
    return text_ft

def macberta_get_feature(text_list, list_len):
    # Download pretrained model from Internet to "\.cache\torch\transformers\"
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-macbert-large')
    bert = BertModel.from_pretrained('hfl/chinese-macbert-large')

    # text_list = ["中文", "自然语言处理"]
    inputs = tokenizer(text_list, return_tensors="pt", padding=True)  # "pt"表示"pytorch"
    outputs = bert(**inputs)

    # print(outputs)
    print(outputs.pooler_output)
    print(outputs.pooler_output.shape)
    text_ft = outputs.pooler_output.detach().numpy().reshape(list_len, -1)
    print(text_ft.shape)
    return text_ft

def get_answer_text(csv_path):
    df = pd.read_csv(csv_path, usecols=[0], encoding='GB18030')
    text_list = df['answer'].tolist()
    # print(text_list)
    return  text_list

def get_average_feature(text_list):
    return np.mean(text_list, axis=0)

if __name__ == '__main__':
    # Local pretrained model
    # bert_path = './chinese_macbert_large'
    # tokenizer = BertTokenizer.from_pretrained(bert_path)
    # bert = BertModel.from_pretrained(bert_path, return_dict=True)

    pert = []
    roberta = []
    electra = []
    macberta = []
    name = ['pert', 'roberta', 'macberta', 'electra']

    root = os.path.dirname(os.getcwd())
    print("root direction", root)

    dataset_class = os.listdir(root + '/Dataset')
    print(dataset_class)

    # 遍历数据集的时间序列
    for sample_class in dataset_class:
        sample_class_path = root + '/Dataset' + '/' + sample_class
        # print(sample_class_path)
        sample_file = os.listdir(sample_class_path)
        # print(sample_file)

        for detail in sample_file:
            detail_path = sample_class_path + '/' + detail
            sample_detail = os.listdir(detail_path)
            print(detail_path)

            for csv_file in sample_detail:
                if 'answer' in csv_file and 'csv' in csv_file:
                    #提取答案到text_list
                    text_list = get_answer_text(detail_path + '/' + csv_file)
                    print(text_list)
                    pert_ft = pert_get_feature(text_list, len(text_list))
                    pert.append(get_average_feature(pert_ft))
                    print(np.array(pert).shape)
                    # print(get_average_feature(pert_ft))
                    roberta_ft = roberta_get_feature(text_list, len(text_list))
                    roberta.append(get_average_feature(roberta_ft))
                    print(np.array(roberta).shape)
                    # print(get_average_feature(roberta_ft))

                    macberta_ft = macberta_get_feature(text_list, len(text_list))
                    macberta.append(get_average_feature(macberta_ft))
                    print(np.array(macberta).shape)
                    # print(get_average_feature(macberta_ft))

                    electra_ft = electra_get_feature(text_list, len(text_list))
                    electra.append(get_average_feature(electra_ft))
                    # print(get_average_feature(electra_ft))
                    print(np.array(electra).shape)

            pert = np.array(pert).reshape(-1, 1)
            roberta = np.array(roberta).reshape(-1, 1)
            macberta = np.array(macberta).reshape(-1, 1)
            electra = np.array(electra).reshape(-1, 1)

            text_sum = np.stack((pert, roberta, macberta, electra), axis=1)
            text_sum = text_sum.reshape(-1, 4)

            # 写入csv
            landmarks = pd.DataFrame(columns=name, data=text_sum)
            # print(landmarks)
            # print(face_x)
            landmarks.to_csv(detail_path + '/text_feature.csv', encoding='gbk')

            text_sum = []
            pert = []
            roberta = []
            macberta = []
            electra = []
