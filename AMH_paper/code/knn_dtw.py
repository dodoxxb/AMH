# -*- coding:utf8 -*-
# @Author  :Dodo

import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

#toy dataset
# X = np.random.random((100,10))
# print(X.shape[1])
# y = np.random.randint(0,2, (100))
# print(y)


# plt.plot(X,y, 'bx')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.show()

#load data
def load_data(csv_path):

    X = []
    df_br = pd.read_csv(csv_path, encoding="gbk", usecols=["br"])
    # print(df_br)

    for index, row in df_br.iterrows():
        X.append(row['br'])

    avg = np.mean(X)

    for i in range(500-index):
        X.append(avg)

    # X.reshape(index,1)
    # print(X)
    # print(type(X))
    return X

#custom metric
def DTW(a, b):
    an = a.size
    bn = b.size
    pointwise_distance = distance.cdist(a.reshape(-1,1),b.reshape(-1,1))
    cumdist = np.matrix(np.ones((an+1,bn+1)) * np.inf)
    cumdist[0,0] = 0

    for ai in range(an):
        for bi in range(bn):
            minimum_cost = np.min([cumdist[ai, bi+1],
                                   cumdist[ai+1, bi],
                                   cumdist[ai, bi]])
            cumdist[ai+1, bi+1] = pointwise_distance[ai,bi] + minimum_cost
    # print(a,b, cumdist[an, bn])
    return cumdist[an, bn]

if __name__ == '__main__':

    X = []
    # print(type(X))
    y = []

    dataset_path = './dataset'
    json_suffix = '.json'
    csv_suffix = '.csv'
    dataset_class = os.listdir(dataset_path)

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
                temp = load_data(csv_path)
                # temp = np.array(temp)
                # print(temp.shape[0])
                X.append(temp)
                y.append(1)
            elif sample_class == 'unhealthy':
                # print('unhealthy')
                # load_data(csv_path)
                # print(load_data(csv_path))
                temp = load_data(csv_path)
                # temp = np.array(temp)
                X.append(temp)
                y.append(0)
            else :
                print('Another data file exist')
                continue


    print(X)
    print(y)

    # print(DTW(X[0],X[1]))
    # X = np.array(X).reshape(-1,1)
    # print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # x_train = np.array(X_train)
    # y_train = np.array(y_train)
    # x_test = np.array(X_test)
    # y_test = np.array(y_test)
    # print(x_train)
    # print(y_train)
    # print(x_test)
    # print(y_test)

    # train
    parameters = {'n_neighbors':[43]}
    clf = GridSearchCV(KNeighborsClassifier(metric =DTW), parameters, cv=5, scoring='accuracy')
    clf.fit(X_train, y_train)

    # # evaluate
    # y_pred = clf.predict(X_test)
    # print(classification_report(y_test, y_pred))
    # print('best_score:', clf.best_score_)
    # print('best parameters:', clf.best_params_)
    # print('best model:', clf.best_estimator_)