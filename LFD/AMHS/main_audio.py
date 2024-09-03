
# imports
import os
import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
import cv2 as cv
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from config import AUDIO_MELSPEC_PATH, CHOSEN_PATH
from models.xception import xception

class AudioNet(torch.nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(AudioNet, self).__init__()
        self.hidden1 = torch.nn.Linear(n_input,n_hidden)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.dropout1(x)
        x = self.hidden2(x)
        x = self.dropout2(x)
        x = self.predict(x)
        return x

def get_chosen_id(path = CHOSEN_PATH):
    df = pd.read_csv(path["healthy"])
    id_pos = list(df["id"])
    df = pd.read_csv(path["unhealthy"])
    id_neg = list(df["id"])

    return id_pos, id_neg

def make_audio_dataset(pos_path, neg_path, sample_id = None):
    dataset_x = []
    dataset_y = []

    if sample_id is None:
        # 没有指定选取数据的范围，表示全部加载
        for root, _, files in os.walk(pos_path):
            if len(_) == 0:
                imgs = []
                for item in files:
                    img = cv.imread(os.path.join(root, item))
                    imgs.append(img)
                if len(imgs) > 0:
                    dataset_x.append(imgs)
                    dataset_y.append(1)
        for root, _, files in os.walk(neg_path):
            if len(_) == 0:
                imgs = []
                for item in files:
                    img = cv.imread(os.path.join(root, item))
                    imgs.append(img)
                if len(imgs) > 0:
                    dataset_x.append(imgs)
                    dataset_y.append(0)
    else:
        id_pos, id_neg = sample_id
        for item in id_pos:
            path = os.path.join(pos_path, item)
            imgs = []
            for root, _, files in os.walk(path):
                for file in files:
                    spec_img = cv.imread(os.path.join(path, file))
                    imgs.append(spec_img)
            if len(imgs)>0:
                dataset_x.append(imgs)
                dataset_y.append(1)

        for item in id_neg:
            path = os.path.join(neg_path, item)
            imgs = []
            for root, _, files in os.walk(path):
                for file in files:
                    spec_img = cv.imread(os.path.join(path, file))
                    imgs.append(spec_img)
            if len(imgs)>0:
                dataset_x.append(imgs)
                dataset_y.append(0)

    return np.array(dataset_x), np.array(dataset_y)

def generate_audio_feature(x):
    features = []
    xception_model = xception(pretrained = True, pretrained_path = "pretrained/xception/xception_torch/xception-c0a72b38.pth.tar")

    xception_model.eval()
    for imgs in x:
        i_feature = []
        for item in imgs:
            item = item / 255.0
            item = torch.from_numpy(cv.resize(item, (299, 299)).reshape((1,3,299,299))).type(torch.float32) # n_samples, channels, height, width
            value = xception_model(item)
            i_feature.append(value.detach().numpy())
        i_feature = np.array(i_feature)
        i_feature = np.mean(i_feature, axis = 0)
        features.append(i_feature)

    return np.array(features)

if __name__ == "__main__":
    # 载入音频数据
    # pos_audio_path = AUDIO_MELSPEC_PATH["healthy"]
    # neg_audio_path = AUDIO_MELSPEC_PATH["unhealthy"]
    # id_pos, id_neg = get_chosen_id()
    # x, y = make_audio_dataset(pos_path = pos_audio_path, neg_path = neg_audio_path, sample_id = (id_pos, id_neg))
    # x = generate_audio_feature(x)
    # np.save("audio_feature_x.npy", x)
    # np.save("audio_y.npy", y)
    # print("finish loading data")
    #=========================================================#
    x = np.load("audio_feature_x.npy")
    y = np.load("audio_y.npy")
    num, _, dim = x.shape
    x = x.reshape((num, dim))

    x_1 = [x[i] for i in range(x.shape[0]) if y[i] == 1]
    x_0 = [x[i] for i in range(x.shape[0]) if y[i] == 0]
    y_1 = [y[i] for i in range(x.shape[0]) if y[i] == 1]
    y_0 = [y[i] for i in range(x.shape[0]) if y[i] == 0]
    sample_number = len(y_0)

    x_1_largetest = np.array(x_1[sample_number:]) # 只会在全数据集测试中会用到的数据
    y_1_largetest = np.array(y_1[sample_number:])
    x_1 = x_1[:sample_number] # 留下用于训练和valid的数据
    y_1 = y_1[:sample_number]

    # train和valid中会用到的x和y
    x = np.vstack((np.array(x_1), np.array(x_0)))
    y = np.hstack((np.array(y_1), np.array(y_0)))

    index = [i for i in range(y.shape[0])]
    random.shuffle(index)
    x = x[index]
    y = y[index]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train, y_train = Variable(torch.from_numpy(x_train).type(torch.FloatTensor)), Variable(torch.from_numpy(y_train).type(torch.LongTensor))
    x_test, y_test = torch.from_numpy(x_test).type(torch.FloatTensor), torch.from_numpy(y_test).type(torch.LongTensor)
    print("=" * 10, "dataset loaded!", "=" * 10)

    # 网络
    audionet = AudioNet(dim, 512, 2)
    print(audionet)

    optimizer = torch.optim.Adam(audionet.parameters(), lr=0.0001)
    loss_func = torch.nn.CrossEntropyLoss()

    # 训练
    epochs = 200
    for t in range(epochs):
        out = audionet(x_train)
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
    audionet.eval()
    out = audionet(x_test)
    prediction = []
    for item in out:
        if item[0] > item[1]:
            prediction.append(0)
        else:
            prediction.append(1)
    pred_y = np.array(prediction)
    target_y = y_test.data.numpy()
    print("test acc %.4f on the 1:1 test set" % (accuracy_score(pred_y, target_y)))
    #
    # 保存
    torch.save(audionet, "audionet_mel.pth")