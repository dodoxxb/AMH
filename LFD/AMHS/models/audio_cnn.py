"""
using mfcc feature and cnn classifier
"""
import os
import torch
import torch.nn as nn
import librosa
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 2 classes
        self.out = nn.Linear(32 * 7 * 7, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x    # return x for visualization

def get_mfcc(file_path, duration = 5, offset = 0.5):
    # 截取特定时间的mfcc，如果不满，则填0
    # 选择duration = 5， 特征维度 20*216
    x, sr = librosa.load(file_path, duration=duration, offset=offset)
    signal = np.zeros(int(sr * duration))
    signal[:len(x)] = x
    mel_spectrogram = librosa.feature.mfcc(signal, sr)
    return mel_spectrogram

def make_dataset_(path, label):
    x = []
    y = []
    for root, folders, files in os.walk(path):
        if len(folders) == 0:
            wav_files = [item for item in files if ".wav" in item]
            mfccs = []
            for file in wav_files:
                mfcc = get_mfcc(os.path.join(root, file))
                mfccs.append(mfcc)
            x.append(np.mean(np.array(mfccs), axis = 0))
            # 虽然我觉得mean不可行
            # 把5个gram vstack然后再cnn好像也不行
            # 5个lstm， concat， fully-connected? have a try.
            # the question is mfcc之后怎么继续做
            y.append(label)
            break
    return np.array(x), np.array(y)

def make_dataset(pos_path, neg_path):
    x_1, y_1 = make_dataset_(pos_path, 1)
    x_0, y_0 = make_dataset_(neg_path, 0)


if __name__ == "__main__":
    x, y = make_dataset_(r"D:\projects\fang\data\data_ok\unhealthy", 0)
    print(x.shape)
    # 原始路径，读入音频，转成mfcc--x，y
    # path = {
    #     "healthy": r"D:\projects\fang\data\data_ok\healthy",
    #     "unhealthy": r"D:\projects\fang\data\data_ok\unhealthy"
    # }
    # x_train, y_train, x_valid, y_valid, x_test_large, y_large = make_dataset(path["healthy"], path["unhealthy"])
    # 划分训练（train+valid）集、大测试集
    # 模型
    # 训练
    # 测试
    # 保存模型
