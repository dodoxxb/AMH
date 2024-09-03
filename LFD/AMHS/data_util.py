# coding = utf-8


# imports
import os
import torch
import cv2 as cv
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from ft_config import *
from timm.models import create_model


class SentenceDataset(Dataset):
    """
        for finetune on a sentence classification task.
        include question in the sentence embedding.
        single sentence level, each answer is treated as one sample.
    """
    def __init__(self, csv_files, tokenizer, train=False):
        """
        notice: negative samples are healthy ones.
        :param csv_file: path of healthy and unhealthy samples' csv file.
        """
        def generate_corpus(df):
            df_sentence = [list(df["ans0"]), list(df["ans1"]), list(df["ans2"]), list(df["ans3"]), list(df["ans4"])]
            topics = ["兴趣:", "偶像:", "崇拜:", "开心:", "困扰:"]
            num_sent = len(df_sentence[0])
            sentences = []
            for i in range(num_sent):
                for topic_idx in range(5):
                    ans = df_sentence[topic_idx][i]
                    if type(ans) == str:
                        sentences.append(topics[topic_idx]+ans)
            return sentences

        unhealthy_sentence = generate_corpus(pd.read_csv(csv_files[1]))
        healthy_sentence = generate_corpus(pd.read_csv(csv_files[0]))

        self.tokenizer = tokenizer
        self.num_neg = len(healthy_sentence)
        self.num_pos = len(unhealthy_sentence)
        if train:
            sample_num = min(self.num_neg, self.num_pos)
            self.num_pos = sample_num
            self.num_neg = sample_num
            self.corpus = healthy_sentence[:sample_num] + unhealthy_sentence[:sample_num]
            self.labels = np.concatenate((np.zeros(sample_num), np.ones(sample_num)))
        else:
            self.corpus = healthy_sentence + unhealthy_sentence
            self.labels = np.concatenate((np.zeros(self.num_neg), np.ones(self.num_pos)))
        self.train = train

    def __len__(self):
        return self.num_pos + self.num_neg

    def __getitem__(self, item):
        text = self.corpus[item]
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > MAX_LEN - 2:
            tokens = tokens[:MAX_LEN - 2]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        segments = [0] * len(input_ids)
        input_masks = [1] * len(input_ids)

        padding_length = MAX_LEN - len(input_ids)
        input_ids = input_ids + [0] * padding_length
        segments = segments + [0] * padding_length
        input_masks = input_masks + [0] * padding_length

        return torch.tensor(input_ids, dtype=torch.long, device=DEVICE), \
               torch.tensor(segments, dtype=torch.long, device=DEVICE), \
               torch.tensor(input_masks, dtype=torch.long, device=DEVICE), \
               torch.tensor(self.labels[item], dtype=torch.long, device=DEVICE)


class TextDataset(Dataset):
    """
        Text Dataset.
        answers from a single student (5 answers) are treated as one sample.
        the length of a single answer is limited to 32. 句子对齐.
    """
    def __init__(self, csv_files, tokenizer, train=False):
        """
        notice: negative samples are healthy ones.
        :param csv_file: path of healthy and unhealthy samples' csv file.
        """
        def generate_corpus(df):
            df_id = list(df["unique_id"])
            df_sentence = [list(df["ans0"]), list(df["ans1"]), list(df["ans2"]), list(df["ans3"]), list(df["ans4"])]
            # topics = ["兴趣:", "偶像:", "崇拜:", "开心:", "困扰:"]
            num_sent = len(df_sentence[0])
            sentences = []
            for i in range(num_sent):
                answer = ""
                for topic_idx in [2]:
                    ans = df_sentence[topic_idx][i]
                    if type(ans) == str:
                        if len(ans) > 32:
                            ans = ans[:32]
                        elif len(ans) < 32:
                            ans += " " * (32 - len(ans))
                        answer += ans  # topics[topic_idx] + ans
                sentences.append(answer)
                # print(len(answer))
            return df_id, sentences

        unhealthy_id, unhealthy_sentence = generate_corpus(pd.read_csv(csv_files[1]))
        healthy_id, healthy_sentence = generate_corpus(pd.read_csv(csv_files[0]))

        self.tokenizer = tokenizer
        self.num_neg = len(healthy_sentence)
        self.num_pos = len(unhealthy_sentence)
        if train:
            sample_num = min(self.num_neg, self.num_pos)
            self.num_pos = sample_num
            self.num_neg = sample_num
            self.corpus = healthy_sentence[:sample_num] + unhealthy_sentence[:sample_num]
            self.ids = healthy_id[:sample_num] + unhealthy_id[:sample_num]
            self.labels = np.concatenate((np.zeros(sample_num), np.ones(sample_num)))
        else:
            self.corpus = healthy_sentence + unhealthy_sentence
            self.ids = healthy_id + unhealthy_id
            self.labels = np.concatenate((np.zeros(self.num_neg), np.ones(self.num_pos)))
        self.train = train

    def __len__(self):
        return self.num_pos + self.num_neg

    def __getitem__(self, item):
        text = self.corpus[item]
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > MAX_LEN - 2:
            tokens = tokens[:MAX_LEN - 2]
        tokens = ["CLS"] + tokens + ["SEP"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        segments = [0] * len(input_ids)
        input_masks = [1] * len(input_ids)

        padding_length = MAX_LEN - len(input_ids)
        input_ids = input_ids + [0] * padding_length
        segments = segments + [0] * padding_length
        input_masks = input_masks + [0] * padding_length

        return torch.tensor(input_ids, dtype=torch.long, device=DEVICE), \
               torch.tensor(segments, dtype=torch.long, device=DEVICE), \
               torch.tensor(input_masks, dtype=torch.long, device=DEVICE), \
               torch.tensor(self.labels[item], dtype=torch.long, device=DEVICE), \
               self.ids[item]


class AudioDataset(Dataset):
    pass


class PictureDataset(Dataset):
    """ dataset for single face image. """
    def __init__(self, pic_path, model, train):
        path_healthy, path_unhealthy = pic_path
        def read_img(path):
            imgs = []
            ids = []
            num = 0
            for root, folder, files in os.walk(path):
                for file in files:
                    img = cv.imread(os.path.join(root, file))
                    img = cv.resize(img, (224, 224), interpolation=cv.INTER_AREA)
                    num += 1
                    imgs.append(img)
                    ids.append(file[:-4])
            return imgs, ids, num

        imgs_healthy, ids_healthy, self.num_neg = read_img(path_healthy)
        imgs_unhealthy, ids_unhealthy, self.num_pos = read_img(path_unhealthy)

        if train:
            num = min(self.num_neg, self.num_pos)
            self.imgs = imgs_healthy[:num] + imgs_unhealthy[:num]
            self.ids = ids_healthy[:num] + ids_unhealthy[:num]
            self.labels = [0] * num + [1] * num
            self.num_pos = num
            self.num_neg = num
        else:
            self.imgs = imgs_healthy + imgs_unhealthy
            self.ids = ids_healthy + ids_unhealthy
            self.labels = [0] * self.num_neg + [1] * self.num_pos
        self.model = model

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        self.model.train(False)
        img = torch.from_numpy(self.imgs[item]) / 255.0
        img = img.reshape((1, 3, 224, 224))
        feature = self.model.forward_features(img)
        return feature, torch.tensor(self.labels[item])

    def generate_feature(self):
        self.model.train(False)
        imgs = torch.from_numpy(np.array(self.imgs)) / 255
        imgs = imgs.reshape((-1, 3, 224, 224))
        features = self.model.forward_features(imgs)

        return self.ids, features, self.labels


class VideoDataset(Dataset):
    pass


class AMHDataset(Dataset):
    pass


if __name__ == "__main__":
    model = create_model(
        "deit_base_patch16_224",
        pretrained=True,
        num_classes=2,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
    )
    dataset = PictureDataset([r"data_preprocessed\single_img\healthy_normalized",
                              r"data_preprocessed\single_img\unhealthy_normalized"], model, False)

    ids, features, labels = dataset.generate_feature()
    features = features.detach().numpy()
    np.savetxt("face_healthy_feature.csv",  features[:-95, :], delimiter=",")
    np.savetxt("face_unhealthy_feature.csv",  features[-95:, :], delimiter=",")
