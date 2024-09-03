
from models.bert import BertTextNet
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertTokenizer

PATH = {
    "pos_path_small": "data_preprocessed/text_healthy_chosen.csv",
    "neg_path_small": "data_preprocessed/text_unhealthy_chosen.csv",
    "pos_path_all": "data_preprocessed/text_healthy.csv",
    "neg_path_all": "data_preprocessed/text_unhealthy.csv"
}

class TextNet(torch.nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(TextNet,self).__init__()
        self.hidden1 = torch.nn.Linear(n_input,n_hidden)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.hidden2 = torch.nn.Linear(n_hidden,n_hidden)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.predict = torch.nn.Linear(n_hidden,n_output)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.dropout1(x)
        x = self.hidden2(x)
        x = self.dropout2(x)
        x = self.predict(x)
        return x

def pooling(vecs, method):
    vecs = np.array(vecs)
    if method == "max": # max方法的效果差的一匹
        return np.max(vecs, axis = 0)
    elif method == "mean":
        return np.mean(vecs, axis = 0)
    else:
        """
        其他融合特征的策略可以写在这里，这里的return vecs[0]没有任何实际意义
        """
        return vecs[0]

def get_embeddings(texts, textnet = None, pretrained_path = "hfl/chinese-roberta-wwm-ext-large"):
    if textnet is None:
        textnet = BertTextNet(pretrained_path)
    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    tokens, segments, input_masks = [], [], []
    for text in texts:
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens.append(indexed_tokens)
        segments.append([0] * len(indexed_tokens))
        input_masks.append([1] * len(indexed_tokens))

    max_len = max([len(single) for single in tokens])
    for j in range(len(tokens)):
        padding = [0] * (max_len - len(tokens[j]))
        tokens[j] += padding
        segments[j] += padding
        input_masks[j] += padding

    tokens_tensor = torch.tensor(tokens)
    segments_tensor = torch.tensor(segments)
    input_mask_tensor = torch.tensor(input_masks)

    text_hashCodes = textnet(tokens_tensor, segments_tensor, input_mask_tensor)

    return text_hashCodes

def make_text_dataset(pos_path, neg_path, pretrained_path):
    df_pos = pd.read_csv(pos_path)
    original_text = [df_pos["ans0"],
                     df_pos["ans1"],
                     df_pos["ans2"],
                     df_pos["ans3"],
                     df_pos["ans4"],
             ]
    dataset_x = []
    dataset_y = []
    textnet = BertTextNet(pretrained_path)
    for i in range(len(df_pos["ans0"])): # 一共有这么多学生
        texts = []
        for j in range(5):
            if isinstance(original_text[j][i], str):
                texts.append("[CLS]" + original_text[j][i] + "[SEP]")
        feature = np.zeros(768)
        if not texts == []:
            vecs = get_embeddings(texts, textnet=textnet, pretrained_path=pretrained_path)
            feature = pooling([vec.detach().numpy() for vec in vecs], method="max")
        dataset_x.append(feature)
        dataset_y.append(1)

    df_neg = pd.read_csv(neg_path)
    original_text = [df_neg["ans0"],
                     df_neg["ans1"],
                     df_neg["ans2"],
                     df_neg["ans3"],
                     df_neg["ans4"],
                     ]

    for i in range(len(df_neg["ans0"])):  # 一共有这么多学生
        texts = []
        for j in range(5):
            if isinstance(original_text[j][i], str):
                texts.append("[CLS]" + original_text[j][i] + "[SEP]")

        feature = np.zeros(768)
        if not texts == []:
            vecs = get_embeddings(texts, textnet=textnet, pretrained_path=pretrained_path)
            feature = pooling([vec.detach().numpy() for vec in vecs], method="mean")
        dataset_x.append(feature)
        dataset_y.append(0)

    return np.array(dataset_x), np.array(dataset_y)

def choose_sample(samples, count):
    index = [i for i in range(len(samples))]
    random.shuffle(index)
    return samples[:count]

if __name__ == "__main__":
    # 载入所有的正负样本的bert向量
    x, y = make_text_dataset(pos_path = PATH["pos_path_small"], neg_path = PATH["neg_path_small"],
                             pretrained_path = "pretrained/bert/chinese_wwm_ext_pytorch")
    print(x)
    print(y.shape)
    # # 打乱数据集
    # index = [i for i in range(y.shape[0])]
    # random.shuffle(index)
    # x = x[index]
    # y = y[index]
    # # 选出来的数据划分train/test
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    # print(x_train.shape)
    # print(x_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)
    # # np.savetxt("text_x_train.csv", x_train)
    # # np.savetxt("text_y_train.csv", y_train)
    # x_train, y_train = Variable(torch.from_numpy(x_train).type(torch.FloatTensor)), Variable(torch.from_numpy(y_train).type(torch.LongTensor))
    # x_test, y_test = torch.from_numpy(x_test).type(torch.FloatTensor), torch.from_numpy(y_test).type(torch.LongTensor)
    # print("="*10, "dataset loaded!", "="*10)
    #
    # # 网络
    # textnet = TextNet(768, 128, 2)
    # print(textnet)
    #
    # optimizer = torch.optim.SGD(textnet.parameters(), lr=0.01)
    # loss_func = torch.nn.CrossEntropyLoss()
    #
    # # 训练
    # epochs = 100
    # for t in range(epochs):
    #     out = textnet(x_train)
    #     loss = loss_func(out, y_train)
    #
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #
    #     if t % 10 == 0:
    #         prediction = []
    #         for item in out:
    #             if item[0] > item[1]:
    #                 prediction.append(0)
    #             else:
    #                 prediction.append(1)
    #         pred_y = np.array(prediction)
    #         target_y = y_train.data.numpy()
    #         print("epoch %d, train acc %.4f"%(t,accuracy_score(pred_y, target_y)))
    #
    # # 测试
    # textnet.eval()
    # out = textnet(x_test)
    # prediction = []
    # for item in out:
    #     if item[0] > item[1]:
    #         prediction.append(0)
    #     else:
    #         prediction.append(1)
    # pred_y = np.array(prediction)
    # target_y = y_test.data.numpy()
    # print("test acc %.4f on the 1:1 test set" % (accuracy_score(pred_y, target_y)))
    #
    # # 所有数据上测试
    # x, y = make_text_dataset(pos_path = PATH["pos_path_all"],
    #                          neg_path = PATH["neg_path_all"],
    #                          pretrained_path = "pretrained/bert/chinese_wwm_ext_pytorch")
    # x, y = torch.from_numpy(x).type(torch.FloatTensor), torch.from_numpy(y).type(torch.LongTensor)
    # out = textnet(x)
    # prediction = []
    # for item in out:
    #     if item[0] > item[1]:
    #         prediction.append(0)
    #     else:
    #         prediction.append(1)
    # pred_y = np.array(prediction)
    # target_y = y.data.numpy()
    # print("test acc %.4f on the all data" % (accuracy_score(pred_y, target_y)))
    #
    # # 保存模型
    # torch.save(textnet, "./text_classifier/textnet.pth")