import os
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn import linear_model
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler 
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

from ast import Sub
from re import subn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class SubNet(nn.Module):
    '''
    The subnetwork that is used in LMF for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3


class TextSubNet(nn.Module):
    '''
    The LSTM-based subnetwork that is used in LMF for text
    '''

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(TextSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1


class LMF(nn.Module):
    '''
    Low-rank Multimodal Fusion
    '''

    def __init__(self, input_dims, hidden_dims, dropouts, output_dim, rank, use_softmax=False):
        '''
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, hidden dims of the sub-networks
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            output_dim - int, specifying the size of output
            rank - int, specifying the size of rank in LMF
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super(LMF, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.audio_in = input_dims[0]
        self.video_in = input_dims[1]
        self.text_in = input_dims[2]
        self.iris_in = input_dims[3]
        self.physical_in = input_dims[4]

        self.audio_hidden = hidden_dims[0]
        self.video_hidden = hidden_dims[1]
        self.text_hidden = hidden_dims[2]
        self.iris_hidden = hidden_dims[3]
        self.physical_hidden = hidden_dims[4]

        # self.text_out= text_out
        self.output_dim = output_dim
        self.rank = rank
        self.use_softmax = use_softmax

        self.audio_prob = dropouts[0]
        self.video_prob = dropouts[1]
        self.text_prob = dropouts[2]
        self.iris_prob = dropouts[3]
        self.physical_prob = dropouts[4]
        self.post_fusion_prob = dropouts[5]

        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        self.text_subnet = SubNet(self.text_in, self.text_hidden, self.text_prob)
        self.iris_subnet = SubNet(self.iris_in, self.iris_hidden, self.iris_prob)
        self.physical_subnet = SubNet(self.physical_in, self.physical_hidden, self.physical_prob)

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.audio_factor = Parameter(torch.Tensor(self.rank, self.audio_hidden + 1, self.output_dim))
        self.video_factor = Parameter(torch.Tensor(self.rank, self.video_hidden + 1, self.output_dim))
        self.text_factor = Parameter(torch.Tensor(self.rank, self.text_hidden + 1, self.output_dim))
        self.iris_factor = Parameter(torch.Tensor(self.rank, self.iris_hidden + 1, self.output_dim))
        self.physical_factor = Parameter(torch.Tensor(self.rank, self.physical_hidden + 1, self.output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        # init teh factors
        xavier_normal_(self.audio_factor)
        xavier_normal_(self.video_factor)
        xavier_normal_(self.text_factor)
        xavier_normal_(self.physical_factor)
        xavier_normal_(self.iris_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, audio_x, video_x, text_x, iris_x, physical_x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        text_h = self.text_subnet(text_x)
        iris_h = self.iris_subnet(iris_x)
        physical_h = self.physical_subnet(physical_x)
        batch_size = audio_h.data.shape[0]

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), audio_h), dim=1)
        _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), video_h), dim=1)
        _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), text_h), dim=1)
        _iris_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), iris_h), dim=1)
        _physical_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), physical_h), dim=1)

        fusion_audio = torch.matmul(_audio_h, self.audio_factor)
        fusion_video = torch.matmul(_video_h, self.video_factor)
        fusion_text = torch.matmul(_text_h, self.text_factor)
        fusion_iris = torch.matmul(_iris_h, self.iris_factor)
        fusion_physical = torch.matmul(_physical_h, self.physical_factor)

        fusion_zy = fusion_audio * fusion_video * fusion_text * fusion_iris * fusion_physical

        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # use linear transformation instead of simple summation, more flexibility
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        # print(output.shape)
        if self.use_softmax:
            output = F.softmax(output)
        return output


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

def get_avg_face_landmarks(x, y, z, face_num):
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

def get_avg_iris_landmarks(leftx, lefty, rightx, righty, iris_num):
    avg_leftx = np.sum(leftx) / iris_num
    avg_lefty = np.sum(lefty) / iris_num
    avg_rightx = np.sum(rightx) / iris_num
    avg_righty = np.sum(righty) / iris_num

    return avg_leftx, avg_lefty, avg_rightx, avg_righty

def get_physical_index(csv_path):
    df = pd.read_csv(csv_path, usecols=[1])
    df_index = df['TSfresh'].values.tolist()

    return df_index

def get_voice_feature(csv_path):
    df = pd.read_csv(csv_path, usecols=[1])
    df_mfcc = df['MFCC'].values.tolist()
    return df_mfcc

def get_text_feature(csv_path):
    df = pd.read_csv(csv_path, usecols=[2])
    df_text = df['roberta'].values.tolist()
    return df_text



if __name__ == '__main__':
    face = []
    iris = []
    physical_index = []
    mfcc_ft = []
    text = []

    labels = []
    # 标签
    label = {b'healthy': 0, b'unhealthy': 1}

    root = os.getcwd()
    print("root direction", root)

    dataset_class = os.listdir(root + '/Dataset')
    print(dataset_class)

    # 遍历数据集的人脸信息
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
                    # 打开csv文件并读取人脸信息
                    print(detail_path + '/' + csv_file)
                    face_x, face_y, face_z, face_num = get_face_landmarks(detail_path + '/' + csv_file)
                    avg_x, avg_y, avg_z = get_avg_face_landmarks(face_x, face_y, face_z, face_num)
                    temp_face = avg_x + avg_y + avg_z
                    face.append(temp_face)
                    
                    # 加上标签
                    if sample_class == 'healthy':
                        labels.append(0)
                    elif sample_class == 'unhealthy':
                        labels.append(1)

                if 'iris_location' in csv_file:
                    print(detail_path + '/' + csv_file)
                    left_x, left_y, right_x, right_y, iris_num = get_iris_landmarks(detail_path + '/' + csv_file)
                    
                    avg_leftx, avg_lefty, avg_rightx, avg_righty = get_avg_iris_landmarks(left_x, left_y, right_x, right_y, iris_num)
                    # temp_iris = avg_leftx + avg_lefty + avg_rightx + avg_righty
                    # print(np.array(temp_iris).shape)
                    iris.append(avg_leftx)
                    iris.append(avg_lefty)
                    iris.append(avg_rightx)
                    iris.append(avg_righty)
                    print(np.array(iris).shape)

                if 'Physical_feature' in csv_file:
                    print(detail_path + '/' + csv_file)
                    temp_index = get_physical_index(detail_path + '/' + csv_file)
                    physical_index.append(temp_index)
                    print(np.array(physical_index).shape)

                if 'Voice_feature' in csv_file:
                    print(detail_path + '/' + csv_file)
                    temp_mfcc = get_voice_feature(detail_path + '/' + csv_file)
                    mfcc_ft.append(temp_mfcc)
                    print(np.array(mfcc_ft).shape)

                if 'text_feature' in csv_file:
                    print(detail_path + '/' + csv_file)
                    temp_text = get_text_feature(detail_path + '/' + csv_file)
                    text.append(temp_text)
                    print(np.array(text).shape)


                
    iris = np.array(iris).reshape(-1, 4)
    print("Face set:{face}".format(face = np.array(face).shape))
    print("Iris set:{iris}".format(iris = np.array(iris).shape))
    print("Physical index set:{index}".format(index = np.array(physical_index).shape))
    print("MFCC set:{mfcc}".format(mfcc = np.array(mfcc_ft).shape))
    print("Text set:{text}".format(text = np.array(text).shape))
    print("Label set:{label}".format(label = np.array(labels).shape))

    face_x_train, face_x_test, face_y_train, face_y_test = \
        train_test_split(face, labels, random_state=0, train_size=0.7)
    iris_x_train, iris_x_test, iris_y_train, iris_y_test = \
        train_test_split(iris, labels, random_state=0, train_size=0.7)
    physical_index_x_train, physical_index_x_test, physical_index_y_train, physical_index_y_test = \
        train_test_split(physical_index, labels, random_state=0, train_size=0.7)
    mfcc_ft_x_train, mfcc_ft_x_test, mfcc_ft_y_train, mfcc_ft_y_test = \
        train_test_split(mfcc_ft, labels, random_state=0, train_size=0.7)
    text_x_train, text_x_test, text_y_train, text_y_test = \
        train_test_split(text, labels, random_state=0, train_size=0.7)

    dtype = torch.FloatTensor

    face_x_train, face_x_test, face_y_train, face_y_test = np.array(face_x_train), np.array(face_x_test),\
        np.array(face_y_train), np.array(face_y_test)
    iris_x_train, iris_x_test, iris_y_train, iris_y_test = np.array(iris_x_train), np.array(iris_x_test),\
        np.array(iris_y_train), np.array(iris_y_test)
    physical_index_x_train, physical_index_x_test, physical_index_y_train, physical_index_y_test = \
        np.array(physical_index_x_train), np.array(physical_index_x_test), \
            np.array(physical_index_y_train), np.array(physical_index_y_test)
    mfcc_ft_x_train, mfcc_ft_x_test, mfcc_ft_y_train, mfcc_ft_y_test = np.array(mfcc_ft_x_train), \
        np.array(mfcc_ft_x_test), np.array(mfcc_ft_y_train), np.array(mfcc_ft_y_test)
    text_x_train, text_x_test, text_y_train, text_y_test = np.array(text_x_train), np.array(text_x_test),\
        np.array(text_y_train), np.array(text_y_test)

    text_x_train, text_y_train = torch.from_numpy(text_x_train).type(dtype), \
        torch.from_numpy(text_y_train).type(dtype)
    mfcc_ft_x_train, mfcc_ft_y_train = torch.from_numpy(mfcc_ft_x_train).type(dtype), \
        torch.from_numpy(mfcc_ft_y_train).type(dtype)
    face_x_train, face_y_train = torch.from_numpy(face_x_train).type(dtype), \
        torch.from_numpy(face_y_train).type(dtype)
    iris_x_train, iris_y_train = torch.from_numpy(iris_x_train).type(dtype), \
        torch.from_numpy(iris_y_train).type(dtype)
    physical_index_x_train, physical_index_y_train = torch.from_numpy(physical_index_x_train).type(dtype), \
        torch.from_numpy(physical_index_y_train).type(dtype)

    text_x_test, text_y_test = torch.from_numpy(text_x_test).type(dtype), \
        torch.from_numpy(text_y_test).type(dtype)
    mfcc_ft_x_test, mfcc_ft_y_test = torch.from_numpy(mfcc_ft_x_test).type(dtype), \
        torch.from_numpy(mfcc_ft_y_test).type(dtype)
    face_x_test, face_y_test = torch.from_numpy(face_x_test).type(dtype), \
        torch.from_numpy(face_y_test).type(dtype)
    iris_x_test, iris_y_test = torch.from_numpy(iris_x_test).type(dtype), \
        torch.from_numpy(iris_y_test).type(dtype)
    physical_index_x_test, physical_index_y_test = torch.from_numpy(physical_index_x_test).type(dtype), \
        torch.from_numpy(physical_index_y_test).type(dtype)

    y_train = text_y_train

    model = LMF(input_dims=(9828, 1404, 1024, 4, 4722), hidden_dims=(64, 64, 64, 64, 64),\
         dropouts=(0.3, 0.3, 0.3, 0.3, 0.3, 0.5), output_dim=1, rank=1)

    loss_func = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters())

    epochs = 10
    y_train = y_train.unsqueeze(-1)

    for t in range(epochs):
        out = model(mfcc_ft_x_train, face_x_train, text_x_train, iris_x_train, physical_index_x_train)
        loss = loss_func(out, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 10 == 9:
            model.train(False)
            pred_y = out.detach().numpy()
            for item in range(pred_y.shape[0]):
                if pred_y[item] > 0.5:
                    pred_y[item] = 1
                else:
                    pred_y[item] = 0
            target_y = y_train.data.numpy()
            print("epoch %d, train acc %.4f" % (t, accuracy_score(pred_y, target_y)))
    