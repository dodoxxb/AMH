import csv
import json
import os
import pandas as pd
import numpy as np

def trans(json_path, csv_path):
    csv_file = open(csv_path, 'w+', newline='')
    answer = []
    question = []
    with open(json_path, encoding="utf-8") as f:
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
    print(data)
    for item in json_dict:
        # print(list(item.values()))
        for i in list(item.values()):
            
            if list(item.values()).index(i) == 1 and len(i) > 0:
                # print(i)
                answer.append(i)
            if list(item.values()).index(i) == 11 and len(i) > 0:
                # print(i)
                question.append(i)
        # data.append(list(item.values()))  # 获取每一行的值value
    
    # print(question)
    # print(answer[answer.index('主动和其他同学说话和交朋友。')])
    
    # if '继续保持这样，一个人也挺好的。' in answer:
    #     del answer[answer.index('继续保持这样，一个人也挺好的。')]
    # if '默不作声的忍受着这一切' in answer:
    #     del answer[answer.index('默不作声的忍受着这一切')]
    # if '主动和其他同学说话和交朋友' in answer:
    #     del answer[answer.index('主动和其他同学说话和交朋友')]
    # if '主动和其他同学说话和交朋友。' in answer:
    #     del answer[answer.index('主动和其他同学说话和交朋友。')]
    del_index = []
    for i in range(len(answer)):
        if '继续保持这样' in answer[i]:
            del_index.append(i)
        if '不管别人的看法' in answer[i]:
            del_index.append(i)
        if '主动和其他同学说话和交朋友' in answer[i]:
            del_index.append(i)
        if '让他们不要再争吵了' in answer[i]:
            del_index.append(i)
        if '默不作声的忍受着这一切' in answer[i]:
            del_index.append(i)
        if '自暴自弃，继续否定自己' in answer[i]:
            del_index.append(i)

    print(del_index)
    del_index.sort(reverse=True)

    for i in del_index:
        del answer[i]

    print(answer)
    csv_file.write("answer\n")
    for line in answer:
        # print(line)
        csv_file.write(str(line)+ "\n")  # 以逗号分隔一行的每个元素，最后换行 fw.close()关闭csv文件

    #关闭文件
    print('Json transfer to CSV finished')
    csv_file.close()

if __name__ == '__main__':
    dataset_class = os.listdir('./muldataset')
    # print(dataset_class)

    for sample_class in dataset_class:
        # if '00004' in sample_class:
        #     continue
        sample_class_path = './muldataset' + '/' + sample_class
        print(sample_class_path)
        sample_file = os.listdir(sample_class_path)
        # print(sample_file)

        for detail in sample_file:
            if 'fq' in detail and 'json' in detail:
                detail_path = sample_class_path + '/' + detail
                # print(detail_path + '/' + json_file)
                print(sample_class_path + '/' + detail, sample_class_path+'/'+'answer.csv')
                trans(sample_class_path + '/' + detail, sample_class_path+'/'+'answer.csv')