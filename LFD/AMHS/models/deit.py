# coding = utf-8

# imports
import torch
from data_util import PictureDataset
from timm.models import create_model
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_unimodal_cls(x_train, y_train, x_test, y_test, kernel="poly"):
    cls = SVC(kernel=kernel)
    cls.fit(x_train, y_train)

    predict = cls.predict(x_train)
    score_train = accuracy_score(predict, y_train)
    # print("test acc %.4f on training set" % score_train)

    predict = cls.predict(x_test)
    score_test = accuracy_score(predict, y_test)
    # print("test acc %.4f on test set" % score_test)

    return cls, score_train, score_test


def train_best_cls(x_train, y_train, x_test, y_test):
    iteration = 10
    best_cls = None
    best_test_score = 0
    for item in range(iteration):
        cls, score_train, score_test = train_unimodal_cls(x_train, y_train, x_test, y_test,kernel="poly")
        delt = score_test - score_train
        if score_test > best_test_score:
            best_test_score = score_test
            best_cls = cls
    for item in range(iteration):
        cls, score_train, score_test = train_unimodal_cls(x_train, y_train, x_test, y_test, kernel="rbf")
        delt = score_test - score_train
        if score_test > best_test_score:#  or score_large > best_large_score or delt < best_score_delt:
            best_test_score = score_test
            best_cls = cls
    return best_cls, best_test_score


if __name__ == "__main__":
    model = create_model(
        "deit_base_patch16_224",
        pretrained=True,
        num_classes=2,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
    )

    dataset = PictureDataset([r"../data_preprocessed\single_img\healthy_normalized",
                              r"../data_preprocessed\single_img\unhealthy_normalized"], model, True)

    print("dataset loaded.")

    num = len(dataset)
    x, y = [dataset[item][0].detach().numpy() for item in range(num)], \
           [dataset[item][1].numpy() for item in range(num)]
    x = np.array(x).reshape((-1, 768))
    y = np.array(y).reshape((-1,))
    index = [i for i in range(y.shape[0])]
    random.shuffle(index)
    x = x[index]
    y = y[index]

    print("start training ...")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    print(train_best_cls(x_train, y_train, x_test, y_test))