# -*- coding:utf8 -*-
# @Author  :Dodo

import csv
import json
import os
import pandas as pd
import numpy as np

def trans(path, csv_path):
    csv_file = open(csv_path, 'w+', newline='')
    with open(path, encoding="utf-8") as f:
        content = f.readlines()

    target = ["\\a", "\\b", "\\e", "\\f", "\\n", "\\v", "\\t", "\\r", "\\w", "\\x", "\\0", "\\2", "\\E", "\\K", "\\Y"]
    replaced = ["/a", "/b", "/e", "/f", "/n", "/v", "/t", "/r", "/w", "/x", "/0", "/2", "/E", "/K", "/Y"]
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
    # print(json_dict)
    data = [list(json_dict[1].keys())]
    # print(data)
    for item in json_dict:

        data.append(list(item.values()))  # 获取每一行的值value
    # print(data)
    for line in data:
        # print(line)
        csv_file.write(str(line)+ "\n")  # 以逗号分隔一行的每个元素，最后换行 fw.close()关闭csv文件

    #关闭文件
    print('Json transfer to CSV finished')
    csv_file.close()

def delete_col(csv_path):
    names = ['age','br','classGrade','createdDate','deltaBp','errorCode','gender','haveFace','hr',
             'hrSnr','hrv','imgBuffer','localPath','missionID','name','officeId','relax','stress',
             'stressSnr','studentID','uniqueStudentId']
    df = pd.read_csv(csv_path, encoding="gbk", header=None, names=names)
    # df.columns = [1]
    # print(df)
    # df.to_csv('./test.csv', index = 0)
    # for column_name in df.columns:
    #     print(column_name)

    # df = pd.read_csv('./test.csv', encoding="gbk")
    # 删除原标题
    df.drop([0], inplace=True)

    df.drop(['age','classGrade','createdDate','errorCode', 'gender', 'haveFace', 'imgBuffer',
             'missionID', 'name', 'officeId', 'studentID', 'uniqueStudentId'], axis=1, inplace=True)
    # print(df)
    print('Delete unnecessary colomn finished')
    df.to_csv(csv_path, index = False)

def delete_row(csv_path):
    df = pd.read_csv(csv_path, encoding="gbk")
    # print(df)
    df_del = pd.read_csv(csv_path, encoding="gbk", usecols=["localPath"])
    # print(csv_path)
    # print(df_del)

    cwd = os.getcwd()
    # print('Current direction:', cwd)

    file_dir = os.path.dirname(csv_path)
    # print('File direction:', file_dir)

    os.chdir(file_dir)

    new_dir = os.getcwd()
    # print('Current direction:', new_dir)

    num = 0
    for index, row in df_del.iterrows():
        # print(index)
        row['localPath'] = row['localPath'].replace('\'', '')
        # print(os.path.exists(row['localPath']))
        if 'png' in row['localPath']:
            # print(row['localPath'])
            strlist = row['localPath'].split('/')
            filename = strlist[-1]
            # print(filename)
            # print(os.path.exists(row['localPath']))

            if os.path.exists(filename):
                # print(filename)
                # print(index)
                num += 1
            else :
                # print(index)
                df.drop(index, inplace=True)
                continue
        else:
            continue

    os.chdir(cwd)
    # print(df)
    # print('sample file number: ', num)
    print('Delete unnecessary row finished')
    df.to_csv(csv_path, index = False)

if __name__ == '__main__':
    # root = os.path.dirname(os.getcwd())
    # print("root direction", root)
    dataset_path = './muldataset'
    # path = './dataset/healthy/00002-0102/00002-0102_emotion.json'  # 获取path参数
    json_suffix = '.json'
    csv_suffix = '.csv'
    # csv_path = path.replace(json_suffix, '') + csv_suffix

    # print(csv_path)
    # trans(path, csv_path)
    # delete_col(csv_path)
    # delete_row(csv_path)

    dataset_class = os.listdir(dataset_path)

    for sample_class in dataset_class:
        if 'label.csv' in sample_class:
            continue
        sample_class_path = dataset_path + '/' + sample_class
        print(sample_class_path)
        sample_file = os.listdir(sample_class_path)
        # print(sample_file)

        for detail in sample_file:
            if 'emotion.csv' in detail or 'label.csv' in detail:
                break
            if 'emotion' in detail and '.json' in detail:
                detail_path = sample_class_path + '/' + detail
                # sample_detail = os.listdir(detail_path)
                # print(sample_detail)
                
                # json_path = detail_path + '/' + detail + '_emotion.json'
                json_path = detail_path
                # print(json_path)
                csv_path = json_path.replace(json_suffix, '') + csv_suffix
                # print(csv_path)
                trans(json_path, csv_path)
                delete_col(csv_path)
                delete_row(csv_path)


