"""
音频数据，单模态，二分类
数据预处理：使用noisereduce降噪，音频数据全部只保留前5秒（不足补0）
使用xlsr预训练模型("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"，已经在中文音频数据集上进行过微调)作为feature extractor
(未用本地训练数据进行fine-tuning)
用简单神经网络(输入层-隐层1-隐层2-输出层)作为分类器
"""
"""
Warning: ugly code.
"""
# imports
import os
import numpy as np
import torch
from torch.autograd import Variable
from transformers import Wav2Vec2Model
import torchaudio
from config import AUDIO_DENOISED_PATH
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random


class AudioNet(torch.nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(AudioNet, self).__init__()
        self.hidden1 = torch.nn.Linear(n_input, n_hidden)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.predict = torch.nn.Linear(n_hidden,n_output)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.dropout1(x)
        x = self.hidden2(x)
        x = self.dropout2(x)
        x = self.predict(x)
        return x


def make_wavfeature_dataset(model_choice, path_dict, feature_path = None, second = 5):
    # model
    model = Wav2Vec2Model.from_pretrained(model_choice)
    model.eval()

    features = []
    labels = []
    path = path_dict["healthy"]
    for root, folders, files in os.walk(path):
        if not len(files) == 0:
            person_feature = []
            for file in files:
                wav_denoised_path = os.path.join(root, file)
                wavform, sr = torchaudio.load(wav_denoised_path)
                #=====修改长度
                wav_std = torch.zeros((1, sr * second))
                if wavform.shape[1] <= sr * second:  # 音频长度不足
                    wav_std[:, :wavform.shape[1]] = wavform
                else:
                    wav_std = wavform[:, : sr * second]
                wavform = wav_std
                #======改好了
                wavform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wavform)
                output = model(wavform)
                # feature = output.last_hidden_state
                feature = output.extract_features
                feature = feature.detach().numpy().squeeze()
                feature = np.mean(feature, axis=0)
                person_feature.append(feature)
            # features.append(np.mean(np.array(person_feature), axis=0))
            # labels.append(1)
            feature = np.mean(np.array(person_feature), axis=0)
            np.save(os.path.join(feature_path["healthy"], root[-10:])+".npy", feature)
            print("finish file:", root)

    path = path_dict["unhealthy"]
    for root, folders, files in os.walk(path):
        if not len(files) == 0:
            person_feature = []
            for file in files:
                wav_denoised_path = os.path.join(root, file)
                wavform, sr = torchaudio.load(wav_denoised_path)
                wav_std = torch.zeros((1, sr * second))
                if wavform.shape[1] <= sr * second:  # 音频长度不足
                    wav_std[:, :wavform.shape[1]] = wavform
                else:
                    wav_std = wavform[:, : sr * second]
                wavform = wav_std
                wavform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wavform)
                output = model(wavform)
                # feature = output.last_hidden_state
                feature = output.extract_features
                feature = feature.detach().numpy().squeeze()
                feature = np.mean(feature, axis=0)
                person_feature.append(feature)
            # features.append(np.mean(np.array(person_feature), axis=0))
            # labels.append(0)
            feature = np.mean(np.array(person_feature), axis=0)
            np.save(os.path.join(feature_path["unhealthy"], root[-10:]) + ".npy", feature)
            print("finish file:", root)
    # return np.array(features), np.array(labels)


def get_wavfeature_dataset(feature_path):
    labels = []
    features = []

    path = feature_path["healthy"]
    for root, folders, files in os.walk(path):
        if not len(files)==0:
            for file in files:
                f = np.load(os.path.join(root,file))
                features.append(f)
                labels.append(1)
    path = feature_path["unhealthy"]
    for root, folders, files in os.walk(path):
        if not len(files) == 0:
            for file in files:
                f = np.load(os.path.join(root,file))
                features.append(f)
                labels.append(0)
    return np.array(features), np.array(labels)


if __name__ == "__main__":
    # try model
    # model_choice = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
    # model = Wav2Vec2Model.from_pretrained(model_choice)
    # model.eval()
    # print(model)

    # =====================================================================================
    # # extract feature and storage( need not to extract again in fusion task
    model_choice = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
    feature_path = {
        "healthy":"data_preprocessed/wav_healthy_feature",
        "unhealthy":"data_preprocessed/wav_unhealthy_feature"
    }
    make_wavfeature_dataset(model_choice, AUDIO_DENOISED_PATH, AUDIO_FEATURE_PATH)
    print("finish")
    """
    # ======================================================================================
    x, y = get_wavfeature_dataset(feature_path)
    # # np.save("wav_features.npy", x)
    # # np.save("wav_labels.npy", y)
    # print(x.shape)
    # print(y.shape)
    # =========================================================================================
    # x = np.load("wav_features.npy")
    # y = np.load("wav_labels.npy")

    # 目前数据集为全体，训练需要在样本正负比为1：1的数据集上进行，所以重新划分
    num, dim = x.shape
    x_1 = [x[i] for i in range(x.shape[0]) if y[i] == 1]
    x_0 = [x[i] for i in range(x.shape[0]) if y[i] == 0]
    y_1 = [y[i] for i in range(x.shape[0]) if y[i] == 1]
    y_0 = [y[i] for i in range(x.shape[0]) if y[i] == 0]
    sample_number = len(y_0)

    x_1_largetest = np.array(x_1[sample_number:])  # 只会在全数据集测试中会用到的数据
    y_1_largetest = np.array(y_1[sample_number:])
    x_1 = x_1[:sample_number]  # 留下用于训练和valid的数据
    y_1 = y_1[:sample_number]

    # train和valid中会用到的x和y
    x = np.vstack((np.array(x_1), np.array(x_0)))
    y = np.hstack((np.array(y_1), np.array(y_0)))

    index = [i for i in range(y.shape[0])]
    random.shuffle(index)
    x = x[index]
    y = y[index]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    print(y_train)
    print(y_test)
    x_train, y_train = Variable(torch.from_numpy(x_train).type(torch.FloatTensor)), Variable(
        torch.from_numpy(y_train).type(torch.LongTensor))
    x_test, y_test = torch.from_numpy(x_test).type(torch.FloatTensor), torch.from_numpy(y_test).type(torch.LongTensor)
    print("=" * 10, "dataset loaded!", "=" * 10)

    # 网络
    audionet = AudioNet(dim, 512, 2)
    print(audionet)

    optimizer = torch.optim.SGD(audionet.parameters(), lr=0.01)
    loss_func = torch.nn.CrossEntropyLoss()

    # 训练
    epochs = 2000
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
    #============================================================================================
    # 1:1 测试
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
    torch.save(audionet, "./audio_classifier/audionet.pth")
    # ===================================================================================================
    # 大数据集测试
    x = np.load("wav_features.npy")
    y = np.load("wav_labels.npy")
    num, dim = x.shape
    x, y = torch.from_numpy(x).type(torch.FloatTensor), torch.from_numpy(y).type(torch.LongTensor)
    audionet.eval()
    out = audionet(x)
    prediction = []
    for item in out:
        if item[0] > item[1]:
            prediction.append(0)
        else:
            prediction.append(1)
    pred_y = np.array(prediction)
    target_y = y.data.numpy()
    print("test acc %.4f on the large test set" % (accuracy_score(pred_y, target_y)))
    """