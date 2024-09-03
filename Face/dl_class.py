from tsai.all import *
import os
import random
import pandas as pd
import numpy as np
from main import get_face_landmarks, get_avg_landmarks

if __name__ == '__main__':
    computer_setup()

    face = []
    labels = []
    # 标签
    label = {b'healthy': 0, b'unhealthy': 1}

    root = os.path.dirname(os.getcwd())
    print("root direction", root)

    dataset_class = os.listdir(root + '/Dataset')
    print(dataset_class)

    # 遍历数据集的人脸信息
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
                if 'face_landmarks' in csv_file:
                    # 打开csv文件并读取人脸信息
                    print(detail_path + '/' + csv_file)
                    face_x, face_y, face_z, face_num = get_face_landmarks(detail_path + '/' + csv_file)
                    avg_x, avg_y, avg_z = get_avg_landmarks(face_x, face_y, face_z, face_num)
                    temp_face = avg_x + avg_y + avg_z
                    # print(np.array(temp_face).shape)
                    face.append(temp_face)

                    # 加上标签
                    if sample_class == 'healthy':
                        labels.append(0)
                    elif sample_class == 'unhealthy':
                        labels.append(1)

    print(np.array(face).shape)
    print(np.array(labels).shape)
