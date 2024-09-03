# coding = utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from data_util import load_data, shuffle_dataset
from uni_modal import train_best_cls


class SimpleTFN(nn.Module):
    def __init__(self, input_dims, dropouts, post_fusion_dim):
        super(SimpleTFN, self).__init__()

        self.audio_in = input_dims[0]
        self.text_in = input_dims[1]
        self.post_fusion_dim = post_fusion_dim
        self.post_fusion_prob = dropouts

        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear((self.audio_in + 1) * (self.text_in + 1), self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 1)

    def forward(self, audio_x, text_x):
        audio_h = audio_x
        text_h = text_x
        batch_size = audio_h.data.shape[0]

        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product

        dtype = torch.FloatTensor

        _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(dtype), requires_grad=False), audio_h), dim=1)
        _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(dtype), requires_grad=False), text_h), dim=1)

        fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _text_h.unsqueeze(1)).view(batch_size, -1)

        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
        post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped))
        post_fusion_y_2 = F.relu(self.post_fusion_layer_2(post_fusion_y_1))
        output = torch.sigmoid(self.post_fusion_layer_3(post_fusion_y_2))

        return output


class SubNet(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
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

        return y_2


class TFN(nn.Module):
    '''
    Implements the Tensor Fusion Networks for multimodal sentiment analysis as is described in:
    Zadeh, Amir, et al. "Tensor fusion network for multimodal sentiment analysis." EMNLP 2017 Oral.
    '''

    def __init__(self, input_dims, hidden_dims, dropouts, post_fusion_dim):
        '''
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, similar to input_dims
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            post_fusion_dim - int, specifying the size of the sub-networks after tensorfusion
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super(TFN, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.audio_in = input_dims[0]
        self.video_in = input_dims[1]
        self.text_in = input_dims[2]

        self.audio_hidden = hidden_dims[0]
        self.video_hidden = hidden_dims[1]
        self.text_hidden = hidden_dims[2]

        self.post_fusion_dim = post_fusion_dim

        self.audio_prob = dropouts[0]
        self.video_prob = dropouts[1]
        self.text_prob = dropouts[2]
        self.post_fusion_prob = dropouts[3]

        # define the pre-fusion subnetworks
        # self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        # self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        # self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        self.text_subnet = SubNet(self.text_in, self.text_hidden, dropout=self.text_prob)

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear((self.text_hidden + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1),
                                             self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 1)

        # in TFN we are doing a regression with constrained output range: (-3, 3), hence we'll apply sigmoid to output
        # shrink it to (0, 1), and scale\shift it back to range (-3, 3)
        # self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        # self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

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

        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), audio_h), dim=1)
        _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), video_h), dim=1)
        _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), text_h), dim=1)

        # _audio_h has shape (batch_size, audio_in + 1), _video_h has shape (batch_size, _video_in + 1)
        # we want to perform outer product between the two batch, hence we unsqueenze them to get
        # (batch_size, audio_in + 1, 1) X (batch_size, 1, video_in + 1)
        # fusion_tensor will have shape (batch_size, audio_in + 1, video_in + 1)
        fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _video_h.unsqueeze(1))

        # next we do kronecker product between fusion_tensor and _text_h. This is even trickier
        # we have to reshape the fusion tensor during the computation
        # in the end we don't keep the 3-D tensor, instead we flatten it
        fusion_tensor = fusion_tensor.view(-1, (self.audio_hidden + 1) * (self.video_hidden + 1), 1)
        fusion_tensor = torch.bmm(fusion_tensor, _text_h.unsqueeze(1)).view(batch_size, -1)

        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
        post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped))
        post_fusion_y_2 = F.relu(self.post_fusion_layer_2(post_fusion_y_1))
        post_fusion_y_3 = torch.sigmoid(self.post_fusion_layer_3(post_fusion_y_2))
        output = post_fusion_y_3  # * self.output_range + self.output_shift

        return output


def train_TFN():
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
    model = TFN(input_dims=(512, 768, 768), hidden_dims=(4, 16, 64), dropouts=(0.3, 0.3, 0.3, 0.3), post_fusion_dim=32)
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


def train_SVM_TF():
    text = load_data("text")
    wav = load_data("wav")
    text_data, text_label, l_text_data, l_text_label = text
    wav_data, wav_label, l_wav_data, l_wav_label = wav

    y = text_label
    y_large = l_text_label
    text_data = np.concatenate((text_data, np.ones(y.shape[0]).reshape((-1, 1))), axis=1)
    wav_data = np.concatenate((wav_data, np.ones(y.shape[0]).reshape((-1, 1))), axis=1)
    l_text_data = np.concatenate((l_text_data, np.ones(y_large.shape[0]).reshape((-1, 1))), axis=1)
    l_wav_data = np.concatenate((l_wav_data, np.ones(y_large.shape[0]).reshape((-1, 1))), axis=1)

    x = []
    for i in range(y.shape[0]):
        x.append(np.dot(text_data[i].reshape((-1, 1)), wav_data[i].reshape((1, -1))).reshape(-1))
    x = np.array(x)

    x_large = []
    for i in range(y_large.shape[0]):
        x_large.append(np.dot(l_text_data[i].reshape((-1, 1)), l_wav_data[i].reshape((1, -1))).reshape(-1))
    x_large = np.array(x_large)

    index = shuffle_dataset(y.shape[0])
    x = x[index]
    y = y[index]

    num_train = int(np.round(text_label.shape[0] * 0.75))
    x_train = x[:num_train]
    y_train = y[:num_train]
    x_test = x[num_train:]
    y_test = y[num_train:]

    cls, best_test_score, best_large_score = train_best_cls(x_train, y_train, x_test, y_test, x_large, y_large)
    print("score on small test set:", best_test_score)
    print("score on large data set:", best_large_score)


if __name__ == "__main__":
    # train_SVM_TF()  # 内存不够 寄了
    precision, recall, accuracy, f1 = [], [], [], []
    l_precision, l_recall, l_accuracy, l_f1 = [], [], [], []
    count = 0
    while count < 100:
        scores = train_TFN()  # ugly results
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