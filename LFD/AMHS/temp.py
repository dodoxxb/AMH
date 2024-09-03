
from models.bert import BertTextNet
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer
import os

def get_embeddings(texts, textnet = None, pretrained_path = "pretrained/bert/chinese_wwm_ext_pytorch"):
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
            feature = pooling([vec.detach().numpy() for vec in vecs], method="mean")
        dataset_x.append(feature)
        dataset_y.append(1)
    return np.array(dataset_x), np.array(dataset_y)


if __name__ == "__main__":
    # path = r"data_preprocessed/text_healthy.csv"
    # df_text = pd.read_csv(text_path)
    # stu_id = df_text["unique_id"]
    # print(len(stu_id))

    # for root, folders, files in os.walk(path_wav):
    #     print(files)
    #     for i in range(len(files)):
    #         print(files[i], " ", stu_id[i])

    # path = r"data_preprocessed\wav_healthy_feature"
    # embedding = []
    # for root, folders, files in os.walk(path):
    #     if not len(files) == 0:
    #         for file in files:
    #             emb = np.load(os.path.join(root, file))
    #             embedding.append(emb)
    # embedding = np.array(embedding)
    # print(embedding.shape)
    # np.savetxt("wav_healthy_feature.csv", embedding, delimiter=",")

    path = r"data_preprocessed/text_healthy.csv"
    pretrained_path="pretrained/bert/chinese_wwm_ext_pytorch"

    df_pos = pd.read_csv(path)
    original_text = [df_pos["ans0"],
                     df_pos["ans1"],
                     df_pos["ans2"],
                     df_pos["ans3"],
                     df_pos["ans4"],
                     ]
    stu_id = df_pos["unique_id"]
    face_path = r"F:\white_fish\projects\fang\AMHS\data_preprocessed\single_img\healthy_normalized"
    for root, folders, files in os.walk(face_path):
        for i in range(len(files)):
            print(files[i], " ", stu_id[i])
    # dataset_x = []
    # textnet = BertTextNet(pretrained_path)
    # for i in range(len(df_pos["ans0"])):  # 一共有这么多学生
    #     texts = []
    #     for j in range(5):
    #         if isinstance(original_text[j][i], str):
    #             texts.append("[CLS]" + original_text[j][i] + "[SEP]")
    #     feature = np.zeros(768)
    #     if not texts == []:
    #         vecs = get_embeddings(texts, textnet=textnet, pretrained_path=pretrained_path)
    #         feature = pooling([vec.detach().numpy() for vec in vecs], method="max")
    #     dataset_x.append(feature)
    #
    # dataset_x = np.array(dataset_x)
    # np.savetxt("text_healthy_feature_max.csv", dataset_x, delimiter=",")