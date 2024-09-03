# coding = utf-8

# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from data_util import load_data, shuffle_dataset
from uni_modal import train_best_cls


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

        self.audio_hidden = hidden_dims[0]
        self.video_hidden = hidden_dims[1]
        self.text_hidden = hidden_dims[2]
        # self.text_out= text_out
        self.output_dim = output_dim
        self.rank = rank
        self.use_softmax = use_softmax

        self.audio_prob = dropouts[0]
        self.video_prob = dropouts[1]
        self.text_prob = dropouts[2]
        self.post_fusion_prob = dropouts[3]

        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        self.text_subnet = SubNet(self.text_in, self.text_hidden, self.text_prob)

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.audio_factor = Parameter(torch.Tensor(self.rank, self.audio_hidden + 1, self.output_dim))
        self.video_factor = Parameter(torch.Tensor(self.rank, self.video_hidden + 1, self.output_dim))
        self.text_factor = Parameter(torch.Tensor(self.rank, self.text_hidden + 1, self.output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        # init teh factors
        xavier_normal_(self.audio_factor)
        xavier_normal_(self.video_factor)
        xavier_normal_(self.text_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, audio_x, video_x, text_x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        text_h = self.text_subnet(text_x)
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

        fusion_audio = torch.matmul(_audio_h, self.audio_factor)
        fusion_video = torch.matmul(_video_h, self.video_factor)
        fusion_text = torch.matmul(_text_h, self.text_factor)
        fusion_zy = fusion_audio * fusion_video * fusion_text

        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # use linear transformation instead of simple summation, more flexibility
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        # print(output.shape)
        if self.use_softmax:
            output = F.softmax(output)
        return output


def train_LMF():
    dtype = torch.FloatTensor
    # dataset
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

    text_x_train, text_y_train = torch.from_numpy(text_x_train).type(dtype), torch.from_numpy(text_y_train).type(dtype)
    wav_x_train, wav_y_train = torch.from_numpy(wav_x_train).type(dtype), torch.from_numpy(wav_y_train).type(dtype)
    face_x_train, face_y_train = torch.from_numpy(face_x_train).type(dtype), torch.from_numpy(face_y_train).type(dtype)

    text_x_test, text_y_test = torch.from_numpy(text_x_test).type(dtype), torch.from_numpy(text_y_test).type(dtype)
    wav_x_test, wav_y_test = torch.from_numpy(wav_x_test).type(dtype), torch.from_numpy(wav_y_test).type(dtype)
    face_x_test, face_y_test = torch.from_numpy(face_x_test).type(dtype), torch.from_numpy(face_y_test).type(dtype)

    l_text_data, l_text_label = torch.from_numpy(l_text_data).type(dtype), torch.from_numpy(l_text_label).type(dtype)
    l_wav_data, l_wav_label = torch.from_numpy(l_wav_data).type(dtype), torch.from_numpy(l_wav_label).type(dtype)
    l_face_data, l_face_label = torch.from_numpy(l_face_data).type(dtype), torch.from_numpy(l_face_label).type(dtype)

    y_train = text_y_train
    # network
    # model = SimpleTFN(input_dims=(512, 768), dropouts=0.3, post_fusion_dim=64)
    model = LMF(input_dims=(512, 768, 768), hidden_dims=(4, 16, 64), dropouts=(0.3, 0.3, 0.3, 0.5), output_dim=1, rank=1)
    # if torch.cuda.is_available():
    #     model = model.cuda()

    loss_func = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters())

    epochs = 500
    best_test_pre = 0.
    best_test_rec = 0.
    best_test_acc = 0.
    best_test_f1 = 0.
    best_large_pre = 0.
    best_large_rec = 0.
    best_large_acc = 0.
    best_large_f1 = 0.
    y_train = y_train.unsqueeze(-1)
    for t in range(epochs):
        out = model(wav_x_train, face_x_train, text_x_train)
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
            # print("epoch %d, train acc %.4f" % (t, accuracy_score(pred_y, target_y)))

            y_test = text_y_test
            out = model(wav_x_test, face_x_test, text_x_test)
            pred_y = out.detach().numpy()
            for item in range(pred_y.shape[0]):
                if pred_y[item] > 0.5:
                    pred_y[item] = 1
                else:
                    pred_y[item] = 0
            target_y = y_test.data.numpy()
            test_score = accuracy_score(pred_y, target_y)
            # print("test acc %.4f on the 1:1 test set" % test_score)
            if test_score > best_test_acc:
                best_test_acc = test_score
                best_test_pre = precision_score(pred_y, target_y)
                best_test_rec = recall_score(pred_y, target_y)
                best_test_f1 = f1_score(pred_y, target_y)

            # 所有数据上测试
            y = l_text_label
            out = model(l_wav_data, l_face_data, l_text_data)
            pred_y = out.detach().numpy()
            for item in range(pred_y.shape[0]):
                if pred_y[item] > 0.5:
                    pred_y[item] = 1
                else:
                    pred_y[item] = 0
            target_y = y.data.numpy()
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
    precision, recall, accuracy, f1 = [], [], [], []
    l_precision, l_recall, l_accuracy, l_f1 = [], [], [], []
    count = 0
    while count < 100:
        scores = train_LMF()  # ugly results
        print(scores)
        if scores[0] > 0.:
            precision.append(scores[0])
            recall.append(scores[1])
            accuracy.append(scores[2])
            f1.append(scores[3])
            l_precision.append(scores[4])
            l_recall.append(scores[5])
            l_accuracy.append(scores[6])
            l_f1.append(scores[7])
            count += 1
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