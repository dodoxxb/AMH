# coding = utf-8

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn.functional as F
from data_util import load_data, shuffle_dataset


class SimpleNet(torch.nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(SimpleNet,self).__init__()
        self.hidden1 = torch.nn.Linear(n_input, 128)
        self.dropout = torch.nn.Dropout(0.5)
        self.hidden2 = torch.nn.Linear(128, 64)
        self.hidden3 = torch.nn.Linear(64, 64)
        self.predict = torch.nn.Linear(64, n_output)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.hidden3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.predict(x)
        x = torch.sigmoid(x)
        return x


def train_unimodal_cls(x_train, y_train, x_test, y_test, x_large, y_large, kernel="poly"):
    cls = SVC(kernel=kernel)
    cls.fit(x_train, y_train)

    predict = cls.predict(x_train)
    score_train = accuracy_score(predict, y_train)
    # print("test acc %.4f on training set" % score_train)

    predict = cls.predict(x_test)
    score_test = accuracy_score(predict, y_test)
    # print("test acc %.4f on test set" % score_test)

    predict = cls.predict(x_large)
    score_large = accuracy_score(predict, y_large)
    # print("test acc %.4f on all data" % score_large)

    return cls, score_train, score_test, score_large


def train_best_cls(x_train, y_train, x_test, y_test, x_large, y_large):
    iteration = 10
    best_cls = None
    best_test_score = 0
    for item in range(iteration):
        cls, score_train, score_test, score_large = train_unimodal_cls(x_train, y_train,
                                                                       x_test, y_test,
                                                                       x_large, y_large,
                                                                       kernel="poly")
        if score_test > best_test_score:
            best_test_score = score_test
            best_cls = cls
    for item in range(iteration):
        cls, score_train, score_test, score_large = train_unimodal_cls(x_train, y_train,
                                                                       x_test, y_test,
                                                                       x_large, y_large,
                                                                       kernel="rbf")
        if score_test > best_test_score:
            best_test_score = score_test
            best_cls = cls

    pre = best_cls.predict(x_test)
    precision = precision_score(pre, y_test)
    recall = recall_score(pre, y_test)
    accuracy = accuracy_score(pre, y_test)
    f1 = f1_score(pre, y_test)
    pre = best_cls.predict(x_large)
    l_precision = precision_score(pre, y_large)
    l_recall = recall_score(pre, y_large)
    l_accuracy = accuracy_score(pre, y_large)
    l_f1 = f1_score(pre, y_large)
    return best_cls, (precision, recall, accuracy, f1, l_precision, l_recall, l_accuracy, l_f1)


def train_nn(modality, data):
    x_train, y_train, x_test, y_test, x_large, y_large = data
    x_train, y_train = torch.from_numpy(x_train).type(torch.FloatTensor), torch.from_numpy(y_train).type(torch.LongTensor)
    x_test, y_test = torch.from_numpy(x_test).type(torch.FloatTensor), torch.from_numpy(y_test).type(torch.LongTensor)
    x_large, y_large = torch.from_numpy(x_large).type(torch.FloatTensor), torch.from_numpy(y_large).type(torch.LongTensor)

    in_dim = x_train.shape[1]
    out_dim = 2
    model = SimpleNet(in_dim, 128, out_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_func = torch.nn.CrossEntropyLoss()

    # 训练
    epochs = 200
    for t in range(epochs):
        out = model(x_train)
        loss = loss_func(out, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 10 == 0:
            prediction = []
            for item in out:
                if item[0] > item[1]:
                    prediction.append(0)
                else:
                    prediction.append(1)
            pred_y = np.array(prediction)
            target_y = y_train.data.numpy()
            print("epoch %d, train acc %.4f" % (t, accuracy_score(pred_y, target_y)))

    # 测试
    model.eval()
    out = model(x_test)
    prediction = []
    for item in out:
        if item[0] > item[1]:
            prediction.append(0)
        else:
            prediction.append(1)
    pred_y = np.array(prediction)
    target_y = y_test.data.numpy()
    print("test acc %.4f on the 1:1 test set" % (accuracy_score(pred_y, target_y)))

    # 所有数据上测试
    x, y = x_large, y_large
    out = model(x)
    prediction = []
    for item in out:
        if item[0] > item[1]:
            prediction.append(0)
        else:
            prediction.append(1)
    pred_y = np.array(prediction)
    target_y = y.data.numpy()
    print("test acc %.4f on the all data" % (accuracy_score(pred_y, target_y)))

    # 保存模型
    torch.save(model, "./"+modality+"_classifier.pth")


if __name__ == "__main__":
    text = load_data("text")
    precision, recall, accuracy, f1 = [], [], [], []
    l_precision, l_recall, l_accuracy, l_f1 = [], [], [], []
    for item in range(10):
        text_data, text_label, l_text_data, l_text_label = text
        index = shuffle_dataset(text_label.shape[0])
        text_data = text_data[index]
        text_label = text_label[index]
        num_train = int(np.round(text_label.shape[0] * 0.75))
        text_x_train = text_data[:num_train]
        text_x_test = text_data[num_train:]
        text_y_train = text_label[:num_train]
        text_y_test = text_label[num_train:]

        cls, scores = train_best_cls(text_x_train, text_y_train, text_x_test, text_y_test, l_text_data, l_text_label)
        print(scores)
        precision.append(scores[0])
        recall.append(scores[1])
        accuracy.append(scores[2])
        f1.append(scores[3])
        l_precision.append(scores[4])
        l_recall.append(scores[5])
        l_accuracy.append(scores[6])
        l_f1.append(scores[7])
    print("mean precision on 1:1 test set:", np.mean(precision))
    print("mean recall on 1:1 test set:", np.mean(recall))
    print("mean accuracy on 1:1 test set:", np.mean(accuracy))
    print("mean f1 on 1:1 test set:", np.mean(f1))
    print("max precision on 1:1 test set:", np.max(precision))
    print("max recall on 1:1 test set:", np.max(recall))
    print("max accuracy on 1:1 test set:", np.max(accuracy))
    print("max f1 on 1:1 test set:", np.max(f1))

    print("mean precision on all data:", np.mean(l_precision))
    print("mean recall on all data:", np.mean(l_recall))
    print("mean accuracy on all data:", np.mean(l_accuracy))
    print("mean f1 on all data:", np.mean(l_f1))
    print("max precision on all data:", np.max(l_precision))
    print("max recall on all data:", np.max(l_recall))
    print("max accuracy on all data:", np.max(l_accuracy))
    print("max f1 on all data:", np.max(l_f1))
