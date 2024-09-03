import os
import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_data(csv_path):
    br = []
    deltaBp = []
    hr = []
    hrSnr = []
    hrv = []
    relax = []
    stress = []
    stressSnr = []


    df_br = pd.read_csv(csv_path, encoding="gbk",
                        usecols=["br", "deltaBp", "hr", "hrSnr", "hrv", "relax", "stress", "stressSnr"])

    for index, row in df_br.iterrows():
        br.append(row['br'])
        deltaBp.append(row['deltaBp'])
        hr.append(row['hr'])
        hrSnr.append(row['hrSnr'])
        hrv.append(row['hrv'])
        relax.append(row['relax'])
        stress.append(row['stress'])
        stressSnr.append(row['stressSnr'])

    avg_br = np.mean(br)
    avg_deltaBp = np.mean(deltaBp)
    avg_hr = np.mean(hr)
    avg_hrSnr = np.mean(hrSnr)
    avg_hrv = np.mean(hrv)
    avg_relax = np.mean(relax)
    avg_stress = np.mean(stress)
    avg_stressSnr = np.mean(stressSnr)

    # print(index)
    for i in range(699-index):
        br.append(avg_br)
        deltaBp.append(avg_deltaBp)
        hr.append(avg_hr)
        hrSnr.append(avg_hrSnr)
        hrv.append(avg_hrv)
        relax.append(avg_relax)
        stress.append(avg_stress)
        stressSnr.append(avg_stressSnr)

    br = np.array(br).reshape(-1,1)
    deltaBp = np.array(deltaBp).reshape(-1,1)
    hr = np.array(hr).reshape(-1,1)
    hrSnr = np.array(hrSnr).reshape(-1,1)
    hrv = np.array(hrv).reshape(-1,1)
    relax = np.array(relax).reshape(-1,1)
    stress = np.array(stress).reshape(-1,1)
    stressSnr = np.array(stressSnr).reshape(-1,1)

    x = np.concatenate( (br,deltaBp,hr,hrSnr,hrv,relax,stress,stressSnr), axis=1 )

    return x

if __name__ == '__main__':
    x = []
    y = []

    dataset_path = './dataset'
    json_suffix = '.json'
    csv_suffix = '.csv'
    dataset_class = os.listdir(dataset_path)

    labels = {0: 'unhealthy', 1: 'healthy'}

    for sample_class in dataset_class:
        sample_class_path = dataset_path + '/' + sample_class
        sample_file = os.listdir(sample_class_path)

        for detail in sample_file:
            detail_path = sample_class_path + '/' + detail
            # print(detail_path)
            sample_detail = os.listdir(detail_path)
            json_path = detail_path + '/' + detail + '_emotion.json'
            # print(json_path)
            csv_path = json_path.replace(json_suffix, '') + csv_suffix
            # print(csv_path)

            if sample_class == 'healthy':
                # print('healthy')
                # load_data(csv_path)
                x.append(load_data(csv_path))
                y.append(1)

            elif sample_class == 'unhealthy':
                x.append(load_data(csv_path))
                y.append(0)

            else :
                print('Another data file exist')
                continue

    x = np.array(x)
    print(x[0])