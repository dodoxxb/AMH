import os
import random

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn import linear_model

def get_face_landmarks(csv_path):
    df = pd.read_csv(csv_path, usecols=[1,2,3])
    # print(df.head(10))
    df_x = df['x']
    df_y = df['y']
    df_z = df['z']
    face_x = df_x.values.tolist()
    face_y = df_y.values.tolist()
    face_z = df_z.values.tolist()

    print("There are {num} faces recorded from this sample.".format(num = np.array(face_x).shape[0] / 468))
    print(np.array(face_x).shape)

    return face_x, face_y, face_z, (np.array(face_x).shape[0] / 468)

def get_avg_landmarks(x, y, z, face_num):
    avg_x = []
    avg_y = []
    avg_z = []

    for i in range(468):
        temp = x[i:len(x):468]
        avg_x.append(np.sum(temp)/face_num)
    for i in range(468):
        temp = y[i:len(x):468]
        avg_y.append(np.sum(temp)/face_num)
    for i in range(468):
        temp = z[i:len(x):468]
        avg_z.append(np.sum(temp)/face_num)

    print(np.array(avg_x).shape)
    return avg_x, avg_y, avg_z

if __name__ == '__main__':
    face = []
    labels = []
    # 标签
    label = {b'healthy':0, b'unhealthy':1}

    root = os.path.dirname(os.getcwd())
    print("root direction", root)

    dataset_class = os.listdir(root + '/Dataset')
    print(dataset_class)

    #遍历数据集的人脸信息
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
                    #打开csv文件并读取人脸信息
                    print(detail_path + '/' + csv_file)
                    face_x, face_y, face_z, face_num = get_face_landmarks(detail_path + '/' + csv_file)
                    # if face_num == 0:
                    #     error = detail
                    #     continue
                    avg_x, avg_y, avg_z = get_avg_landmarks(face_x, face_y, face_z, face_num)
                    temp_face = avg_x + avg_y + avg_z
                    # print(np.array(temp_face).shape)
                    face.append(temp_face)

                    #加上标签
                    if sample_class == 'healthy':
                        labels.append(0)
                    elif sample_class == 'unhealthy':
                        labels.append(1)

                    # face = face.reshape(-1, 1404)
                    # print(np.array(face).shape)

    # print(face)
    print(np.array(face).shape)
    # print(labels)
    print(np.array(labels).shape)
    # print(error)

    # print(np.any(np.isnan(face)))

    # 分类器
    x_train, x_test, y_train, y_test = train_test_split(face, labels, random_state=0, train_size=0.7)
    # print(np.array(x_train).shape)
    print(y_test)

    # LR
    logreg = linear_model.LogisticRegression(max_iter=50000, random_state=0)
    logreg.fit(x_train, y_train)
    score_lr = logreg.score(x_test, y_test)
    y_predicted = logreg.predict(x_test)
    print(y_predicted)
    print("The score of LR is : %f" % score_lr)
    print(metrics.classification_report(y_test, y_predicted))
    print("The weighted f1 score: ", f1_score(y_test, y_predicted, average='weighted'))
    print(metrics.confusion_matrix(y_test, y_predicted))
    scores = cross_val_score(logreg, x_test, y_test, cv=10)
    print('评分：', scores)
    print('准确度：', metrics.accuracy_score(y_predicted, y_test))  # 计算评分的均值

    # kernel = 'rbf'
    clf_rbf = svm.SVC(kernel='rbf')
    clf_rbf.fit(x_train, y_train)
    score_rbf = clf_rbf.score(x_test, y_test)
    y_predicted = clf_rbf.predict(x_test)
    print(y_predicted)
    print("The score of rbf is : %f" % score_rbf)
    print(metrics.classification_report(y_test, y_predicted))
    print("The weighted f1 score: ", f1_score(y_test, y_predicted, average='weighted'))
    print(metrics.confusion_matrix(y_test, y_predicted))
    scores = cross_val_score(clf_rbf, x_test, y_test, cv=10)
    print('评分：', scores)
    print('准确度：', metrics.accuracy_score(y_predicted, y_test))  # 计算评分的均值

    # random forest
    RF_clf = RandomForestClassifier(random_state=0)
    RF_clf.fit(x_train, y_train)
    score_RF = RF_clf.score(x_test, y_test)
    y_predicted = RF_clf.predict(x_test)
    print(y_predicted)
    print("The score of random forest is : %f" % score_RF)
    print(metrics.classification_report(y_test, y_predicted))
    print("The weighted f1 score: ", f1_score(y_test, y_predicted, average='weighted'))
    print(metrics.confusion_matrix(y_test, y_predicted))
    scores = cross_val_score(RF_clf, x_test, y_test, cv=10)
    print('评分：', scores)
    print('准确度：', metrics.accuracy_score(y_predicted, y_test))  # 计算评分的均值

    # NN
    mlp_clf = MLPClassifier(random_state=0, max_iter=5000, hidden_layer_sizes=(4,))
    mlp_clf.fit(x_train, y_train)
    score_nn = mlp_clf.score(x_test, y_test)
    y_predicted = mlp_clf.predict(x_test)
    print(y_predicted)
    print("The score of NN is : %f" % score_nn)
    print(metrics.classification_report(y_test, y_predicted))
    print("The weighted f1 score: ", f1_score(y_test, y_predicted, average='weighted'))
    print(metrics.confusion_matrix(y_test, y_predicted))
    scores = cross_val_score(mlp_clf, x_test, y_test, cv=10)
    print('评分：', scores)
    print('准确度：', metrics.accuracy_score(y_predicted, y_test))  # 计算评分的均值

    # x_train = []
    # y_train = []
    # x_test = []
    # y_test = []
    #
    # random_num = random.sample(range(0,321), 95)
    # print(random_num)
    # for i in range(0,416):
    #     if i >= 321:
    #         x_train.append(face[i])
    #         y_train.append(labels[i])
    #     elif i in random_num:
    #         x_train.append(face[i])
    #         y_train.append(labels[i])
    #     else:
    #         x_test.append(face[i])
    #         y_test.append(labels[i])
    #
    #
    # print(np.array(x_train).shape)
    # print(np.array(y_train).shape)
    # print(np.array(x_test).shape)
    # print(np.array(y_test).shape)
    #
    # # LR
    # logreg = linear_model.LogisticRegression(max_iter=50000, random_state=0)
    # logreg.fit(x_train, y_train)
    # score_lr = logreg.score(x_test, y_test)
    # y_predicted = logreg.predict(x_test)
    # print(y_predicted)
    # print("The score of LR is : %f" % score_lr)
    # print(metrics.classification_report(y_test, y_predicted))
    # print("The weighted f1 score: ", f1_score(y_test, y_predicted, average='weighted'))
    # print(metrics.confusion_matrix(y_test, y_predicted))
    # scores = cross_val_score(logreg, x_test, y_test, cv=10)
    # print('评分：', scores)
    # print('准确度：', metrics.accuracy_score(y_predicted, y_test))  # 计算评分的均值
    #
    # # kernel = 'rbf'
    # clf_rbf = svm.SVC(kernel='rbf')
    # clf_rbf.fit(x_train, y_train)
    # score_rbf = clf_rbf.score(x_test, y_test)
    # y_predicted = clf_rbf.predict(x_test)
    # print(y_predicted)
    # print("The score of rbf is : %f" % score_rbf)
    # print(metrics.classification_report(y_test, y_predicted))
    # print("The weighted f1 score: ", f1_score(y_test, y_predicted, average='weighted'))
    # print(metrics.confusion_matrix(y_test, y_predicted))
    # scores = cross_val_score(clf_rbf, x_test, y_test, cv=10)
    # print('评分：', scores)
    # print('准确度：', metrics.accuracy_score(y_predicted, y_test))  # 计算评分的均值
    #
    # # random forest
    # RF_clf = RandomForestClassifier(random_state=0)
    # RF_clf.fit(x_train, y_train)
    # score_RF = RF_clf.score(x_test, y_test)
    # y_predicted = RF_clf.predict(x_test)
    # print(y_predicted)
    # print("The score of random forest is : %f" % score_RF)
    # print(metrics.classification_report(y_test, y_predicted))
    # print("The weighted f1 score: ", f1_score(y_test, y_predicted, average='weighted'))
    # print(metrics.confusion_matrix(y_test, y_predicted))
    # scores = cross_val_score(RF_clf, x_test, y_test, cv=10)
    # print('评分：', scores)
    # print('准确度：', metrics.accuracy_score(y_predicted, y_test))  # 计算评分的均值
    #
    # # NN
    # mlp_clf = MLPClassifier(random_state=0, max_iter=5000, hidden_layer_sizes=(4,))
    # mlp_clf.fit(x_train, y_train)
    # score_nn = mlp_clf.score(x_test, y_test)
    # y_predicted = mlp_clf.predict(x_test)
    # print(y_predicted)
    # print("The score of NN is : %f" % score_nn)
    # print(metrics.classification_report(y_test, y_predicted))
    # print("The weighted f1 score: ", f1_score(y_test, y_predicted, average='weighted'))
    # print(metrics.confusion_matrix(y_test, y_predicted))
    # scores = cross_val_score(mlp_clf, x_test, y_test, cv=10)
    # print('评分：', scores)
    # print('准确度：', metrics.accuracy_score(y_predicted, y_test))  # 计算评分的均值