import os
import re
import pandas as pd
import numpy as np
from tsfresh.feature_extraction import extract_features
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

def get_index(csv_path):
    df = pd.read_csv(csv_path, usecols=[0,1,2,3,4,6,7,8])

    df_br = df['br']
    df_deltabp = df['deltaBp']
    df_hr = df['hr']
    hrsnr = df['hrSnr'].tolist()
    df_hrv = df['hrv']
    df_relax = df['relax']
    df_stress = df['stress']
    stresssnr = df['stressSnr'].tolist()
    br = df_br.tolist()
    deltabp = df_deltabp.tolist()
    hr = df_hr.tolist()
    hrv = df_hrv.tolist()
    relax = df_relax.tolist()
    stress = df_stress.tolist()


    # print(len(br))
    df.insert(df.shape[1], 'id', 1)
    df.insert(df.shape[1], 'time', np.arange(len(br)))

    # print(df)
    # ft = extract_features(df, column_id="id", column_sort="time")
    # np.nan_to_num(ft)
    # print(np.array(ft).reshape(-1, 2))

    return br, deltabp, hr, hrv, relax, stress, hrsnr, stresssnr

if __name__ == '__main__':
    index_ft = []
    labels = []
    # deltabp = []
    # hr = []
    # hrv = []
    # relax = []
    # stress = []
    # hrsnr = []
    # stresssnr = []
    name = ['br', 'deltabp', 'hr', 'hrv', 'relax', 'stress', 'hrsnr', 'stresssnr']
    # 标签
    label = {b'healthy': 0, b'unhealthy': 1}

    root = os.path.dirname(os.getcwd())
    print("root direction", root)

    dataset_class = os.listdir(root + '/Dataset')
    print(dataset_class)



    #读取时间序列
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
                if 'emotion' in csv_file and 'csv' in csv_file:
                    br, deltabp, hr, hrv, relax, stress, hrsnr, stresssnr = get_index(detail_path + '/' + csv_file)

                else:
                    continue

            for i in range(0, 684 - len(hr)):
                br.append(np.mean(br))
                deltabp.append(np.mean(deltabp))
                hr.append(np.mean(hr))
                hrv.append(np.mean(hrv))
                relax.append(np.mean(relax))
                stress.append(np.mean(stress))
                hrsnr.append(np.mean(hrsnr))
                stresssnr.append(np.mean(stresssnr))


            print(np.array(deltabp).shape)
            br = np.array(br).reshape(-1, 1)
            deltabp = np.array(deltabp).reshape(-1, 1)
            hr = np.array(hr).reshape(-1, 1)
            hrv = np.array(hrv).reshape(-1, 1)
            relax = np.array(relax).reshape(-1, 1)
            stress = np.array(stress).reshape(-1, 1)
            hrsnr = np.array(hrsnr).reshape(-1, 1)
            stresssnr = np.array(stresssnr).reshape(-1, 1)
            print(deltabp.shape)

            physical = np.stack((br, deltabp, hr, hrv, relax, stress, hrsnr, stresssnr), axis=1)
            # print(np.array(physical).shape)
            physical = physical.reshape(-1, 8)
            # print(np.array(physical).shape)
            # 写入csv
            landmarks = pd.DataFrame(columns=name, data=physical)
            # print(landmarks)
            # print(landmarks)
            # print(face_x)
            landmarks.to_csv(detail_path + '/physical_indicators.csv', encoding='gbk')

            # 情况landmarks列表
            physical = []
            br = []
            deltabp = []
            hr = []
            hrv = []
            relax = []
            stress = []
            hrsnr = []
            stresssnr = []

