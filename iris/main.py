import os
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
from imblearn.over_sampling import RandomOverSampler

def get_iris_landmarks(csv_path):
    df = pd.read_csv(csv_path, usecols=[1,2,3,4])
    # print(df.head(10))
    df_leftx = df['left_x']
    df_lefty = df['left_y']
    df_rightx = df['right_x']
    df_righty = df['right_y']
    iris_left_x = df_leftx.values.tolist()
    iris_left_y = df_lefty.values.tolist()
    iris_right_x = df_rightx.values.tolist()
    iris_right_y = df_righty.values.tolist()

    print("There are {num} irises recorded from this sample.".format(num = np.array(df_leftx).shape[0]))
    print(np.array(iris_left_x).shape)

    return iris_left_x, iris_left_y, iris_right_x, iris_right_y, (np.array(df_leftx).shape[0])

def get_avg_landmarks(leftx, lefty, rightx, righty, iris_num):
    avg_leftx = np.sum(leftx) / iris_num
    avg_lefty = np.sum(lefty) / iris_num
    avg_rightx = np.sum(rightx) / iris_num
    avg_righty = np.sum(righty) / iris_num

    return avg_leftx, avg_lefty, avg_rightx, avg_righty

if __name__ == '__main__':
    iris = []
    labels = []
    # 标签
    label = {b'healthy':0, b'unhealthy':1}

    root = os.path.dirname(os.getcwd())
    print("root direction", root)

    dataset_class = os.listdir(root + '/Dataset')
    print(dataset_class)

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
                if 'iris_location' in csv_file:
                    #打开csv文件并读取人脸信息
                    print(detail_path + '/' + csv_file)
                    left_x, left_y, right_x, right_y, iris_num = get_iris_landmarks(detail_path + '/' + csv_file)
                    # if face_num == 0:
                    #     error = detail
                    #     continue
                    avg_leftx, avg_lefty, avg_rightx, avg_righty = get_avg_landmarks(left_x, left_y, right_x, right_y, iris_num)
                    # temp_iris = avg_leftx + avg_lefty + avg_rightx + avg_righty
                    # print(np.array(temp_iris).shape)
                    iris.append(avg_leftx)
                    iris.append(avg_lefty)
                    iris.append(avg_rightx)
                    iris.append(avg_righty)
                    print(np.array(iris).shape)

                    #加上标签
                    if sample_class == 'healthy':
                        labels.append(0)
                    elif sample_class == 'unhealthy':
                        labels.append(1)

                    # face = face.reshape(-1, 1404)
                    # print(np.array(face).shape)

    iris = np.array(iris).reshape(-1, 4)
    print(iris)
    print(np.array(iris).shape)
    # print(labels)
    print(np.array(labels).shape)

    x_train, x_test, y_train, y_test = train_test_split(iris, labels, random_state=0, train_size=0.7)
    # print(np.array(x_train).shape)
    print(y_test)

    cw = {0:3, 1:1}

    # LR
    logreg = linear_model.LogisticRegression(max_iter=50000, random_state=0, class_weight=cw)
    logreg.fit(x_train, y_train)
    score_lr = logreg.score(x_test, y_test)
    y_predicted = logreg.predict(x_test)
    print(y_predicted)
    print("The score of LR is : %f" % score_lr)
    print(metrics.classification_report(y_test, y_predicted))
    print("The weighted f1 score: ", f1_score(y_test, y_predicted, average='weighted'))
    print(metrics.confusion_matrix(y_test, y_predicted))

    print("LR 10-fold:")
    scores = cross_val_score(logreg, iris, labels, cv=10)
    print('评分：', scores)
    print(np.sum(scores) / 10)

    # kernel = 'rbf'
    clf_rbf = svm.SVC(kernel='rbf', class_weight=cw)
    clf_rbf.fit(x_train, y_train)
    score_rbf = clf_rbf.score(x_test, y_test)
    y_predicted = clf_rbf.predict(x_test)
    print(y_predicted)
    print("The score of SVM is : %f" % score_rbf)
    print(metrics.classification_report(y_test, y_predicted))
    print("The weighted f1 score: ", f1_score(y_test, y_predicted, average='weighted'))
    print(metrics.confusion_matrix(y_test, y_predicted))

    print("SVM 10-fold:")
    scores = cross_val_score(clf_rbf, iris, labels, cv=10)
    print('评分：', scores)
    print(np.sum(scores) / 10)

    # random forest
    RF_clf = RandomForestClassifier(random_state=0, class_weight=cw)
    RF_clf.fit(x_train, y_train)
    score_RF = RF_clf.score(x_test, y_test)
    y_predicted = RF_clf.predict(x_test)
    print(y_predicted)
    print("The score of random forest is : %f" % score_RF)
    print(metrics.classification_report(y_test, y_predicted))
    print("The weighted f1 score: ", f1_score(y_test, y_predicted, average='weighted'))
    print(metrics.confusion_matrix(y_test, y_predicted))

    print("RF 10-fold:")
    scores = cross_val_score(RF_clf, iris, labels, cv=10)
    print('评分：', scores)
    print(np.sum(scores) / 10)

    # NN
    mlp_clf = MLPClassifier(random_state=0, max_iter=5000, hidden_layer_sizes=(200,100), shuffle=True)
    mlp_clf.fit(x_train, y_train)
    score_nn = mlp_clf.score(x_test, y_test)
    y_predicted = mlp_clf.predict(x_test)
    print(y_predicted)
    print("The score of NN is : %f" % score_nn)
    print(metrics.classification_report(y_test, y_predicted))
    print("The weighted f1 score: ", f1_score(y_test, y_predicted, average='weighted'))
    print(metrics.confusion_matrix(y_test, y_predicted))

    print("NN 10-fold:")
    scores = cross_val_score(mlp_clf, iris, labels, cv=10)
    print('评分：', scores)
    print(np.sum(scores) / 10)
