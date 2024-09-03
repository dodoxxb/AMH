# coding = utf-8

# imports
import numpy as np
import torch
import torch.nn as nn
from embracenet_pytorch import EmbraceNet
from data_util import load_data, shuffle_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class EmbraceTextAudio(nn.Module):

    def __init__(self, device, input_size, embrace_size):
        super(EmbraceTextAudio, self).__init__()

        self.device = device
        # embracenet
        self.embracenet = EmbraceNet(device=self.device, input_size_list=input_size, embracement_size=embrace_size)

        # post embracement layers
        self.post = nn.Linear(in_features=embrace_size, out_features=2)

    def forward(self, text, wav):
        embraced_output = self.embracenet(input_list=[text, wav])
        x = self.post(embraced_output)
        x = torch.sigmoid(x)
        return x


class EmbraceTextAudioVideo(nn.Module):
    def __init__(self, device, input_size, embrace_size):
        super(EmbraceTextAudioVideo, self).__init__()

        self.device = device
        # embracenet
        self.embracenet = EmbraceNet(device=self.device, input_size_list=input_size, embracement_size=embrace_size)

        # post embracement layers
        self.post = nn.Linear(in_features=embrace_size, out_features=2)

    def forward(self, text, wav, face):
        embraced_output = self.embracenet(input_list=[text, wav, face])
        x = self.post(embraced_output)
        x = torch.sigmoid(x)
        return x


def train_embrace():
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
        ltype = torch.cuda.LongTensor
    else:
        dtype = torch.FloatTensor
        ltype = torch.LongTensor
    # load data
    text = load_data("text")
    wav = load_data("wav")
    face = load_data("face")
    text_data, text_label, l_text_data, l_text_label = text
    wav_data, wav_label, l_wav_data, l_wav_label = wav
    face_data, face_label, l_face_data, l_face_label = face

    index = shuffle_dataset(text_label.shape[0])
    text_data = text_data[index]
    text_label = text_label[index]
    wav_data = wav_data[index]
    wav_label = wav_label[index]
    face_data = face_data[index]
    face_label = face_label[index]

    num_train = int(np.round(text_label.shape[0] * 0.75))

    text_x_train = text_data[:num_train]
    text_x_test = text_data[num_train:]
    text_y_train = text_label[:num_train]
    text_y_test = text_label[num_train:]
    wav_x_train = wav_data[:num_train]
    wav_x_test = wav_data[num_train:]
    wav_y_train = wav_label[:num_train]
    wav_y_test = wav_label[num_train:]
    face_x_train = face_data[:num_train]
    face_x_test = face_data[num_train:]
    face_y_train = face_label[:num_train]
    face_y_test = face_label[num_train:]

    text_x_train, text_y_train = torch.from_numpy(text_x_train).type(dtype), torch.from_numpy(text_y_train).type(ltype)
    wav_x_train, wav_y_train = torch.from_numpy(wav_x_train).type(dtype), torch.from_numpy(wav_y_train).type(ltype)
    face_x_train, face_y_train = torch.from_numpy(face_x_train).type(dtype), torch.from_numpy(face_y_train).type(ltype)

    text_x_test, text_y_test = torch.from_numpy(text_x_test).type(dtype), torch.from_numpy(text_y_test).type(ltype)
    wav_x_test, wav_y_test = torch.from_numpy(wav_x_test).type(dtype), torch.from_numpy(wav_y_test).type(ltype)
    face_x_test, face_y_test = torch.from_numpy(face_x_test).type(dtype), torch.from_numpy(face_y_test).type(ltype)

    l_text_data, l_text_label = torch.from_numpy(l_text_data).type(dtype), torch.from_numpy(l_text_label).type(ltype)
    l_wav_data, l_wav_label = torch.from_numpy(l_wav_data).type(dtype), torch.from_numpy(l_wav_label).type(ltype)
    l_face_data, l_face_label = torch.from_numpy(l_face_data).type(dtype), torch.from_numpy(l_face_label).type(ltype)

    # network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = EmbraceTextAudio(device, [768, 512], 256)
    model = EmbraceTextAudioVideo(device, [768, 512, 768], 256)
    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_func = torch.nn.CrossEntropyLoss()

    # 训练
    y_train = text_y_train
    epochs = 300
    best_test_pre = 0.
    best_test_rec = 0.
    best_test_acc = 0.
    best_test_f1 = 0.
    best_large_pre = 0.
    best_large_rec = 0.
    best_large_acc = 0.
    best_large_f1 = 0.
    for t in range(epochs):
        out = model(text_x_train, wav_x_train, face_x_train)
        loss = loss_func(out, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 10 == 0:
            model.train(False)
            prediction = []
            for item in out:
                if item[0] > item[1]:
                    prediction.append(0)
                else:
                    prediction.append(1)
            pred_y = np.array(prediction)
            target_y = y_train.cpu().data.numpy()
            # print("epoch %d, train acc %.4f" % (t, accuracy_score(pred_y, target_y)))

            # test
            y_test = text_y_test
            out = model(text_x_test, wav_x_test, face_x_test)
            prediction = []
            for item in out:
                if item[0] > item[1]:
                    prediction.append(0)
                else:
                    prediction.append(1)
            pred_y = np.array(prediction)
            target_y = y_test.cpu().data.numpy()
            test_score = accuracy_score(pred_y, target_y)
            # print("test acc %.4f on the 1:1 test set" % test_score)
            if test_score > best_test_acc:
                best_test_acc = test_score
                best_test_pre = precision_score(pred_y, target_y)
                best_test_rec = recall_score(pred_y, target_y)
                best_test_f1 = f1_score(pred_y, target_y)

            # 所有数据上测试
            y = l_text_label
            out = model(l_text_data, l_wav_data, l_face_data)
            prediction = []
            for item in out:
                if item[0] > item[1]:
                    prediction.append(0)
                else:
                    prediction.append(1)
            pred_y = np.array(prediction)
            target_y = y.cpu().data.numpy()
            large_score = accuracy_score(pred_y, target_y)
            # print("test acc %.4f on the all data" % large_score)

            if large_score > best_large_acc:
                best_large_acc = large_score
                best_large_pre = precision_score(pred_y, target_y)
                best_large_rec = recall_score(pred_y, target_y)
                best_large_f1 = f1_score(pred_y, target_y)

            model.train(True)
    return best_test_pre, best_test_rec, best_test_acc, best_test_f1, \
           best_large_pre, best_large_rec, best_large_acc, best_large_f1


if __name__ == "__main__":
    iteration = 10
    precision, recall, accuracy, f1 = [], [], [], []
    l_precision, l_recall, l_accuracy, l_f1 = [], [], [], []
    for i in range(iteration):
        print(">>>in iteration %d :"%i)
        scores = train_embrace()
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