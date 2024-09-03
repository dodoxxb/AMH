# coding = utf-8


# imports
import os
import torch
import torchaudio
import cv2 as cv
from transformers import Wav2Vec2Model
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from audio_config import *


class AudioDataset(Dataset):
    """
        audio Dataset.
        answers from a single student (5 answers) are treated as one sample.
    """

    def __init__(self, text_csv_files=TEXT_CSV_FILES, wav_path=AUDIO_DENOISED_PATH, train=False):
        """
        notice: negative samples are healthy ones.
        :param csv_file: path of healthy and unhealthy samples' csv file.
        """
        def generate_corpus(df):
            df_id = list(df["unique_id"])
            leng = len(list(df["path0"]))
            df_wavs = [list(df["path0"]), list(df["path1"]), list(df["path2"]), list(df["path3"]), list(df["path4"])]
            wav_files = []
            for i in range(leng):
                wav_files.append([df_wavs[0][i][27:], df_wavs[1][i][27:], df_wavs[2][i][27:], df_wavs[3][i][27:], df_wavs[4][i][27:]])
            return df_id, wav_files

        unhealthy_id, unhealthy_wavs = generate_corpus(pd.read_csv(text_csv_files["unhealthy"]))
        healthy_id, healthy_wavs = generate_corpus(pd.read_csv(text_csv_files["healthy"]))

        self.num_neg = len(healthy_id)
        self.num_pos = len(unhealthy_id)
        if train:
            sample_num = min(self.num_neg, self.num_pos)
            self.num_pos = sample_num
            self.num_neg = sample_num
            self.corpus = healthy_wavs[:sample_num] + unhealthy_wavs[:sample_num]
            self.ids = healthy_id[:sample_num] + unhealthy_id[:sample_num]
            self.labels = np.concatenate((np.zeros(sample_num), np.ones(sample_num)))
        else:
            self.corpus = healthy_wavs + unhealthy_wavs
            self.ids = healthy_id + unhealthy_id
            self.labels = np.concatenate((np.zeros(self.num_neg), np.ones(self.num_pos)))
        self.train = train
        self.model = Wav2Vec2Model.from_pretrained(MODEL_CHOICE)
        self.model.eval()

    def __len__(self):
        return self.num_pos + self.num_neg

    def __getitem__(self, item):
        tag = "healthy" if self.labels[item] == 0 else "unhealthy"
        root = "../data_preprocessed/wav_" + tag + "/" + self.ids[item]
        person_feature = []
        for _, __, filenames in os.walk(root):
            for filename in filenames:
                filepath = os.path.join(root, filename)
                wavform, sr = torchaudio.load(filepath)

                wav_std = torch.zeros((1, sr * WAV_SECOND))
                if wavform.shape[1] <= sr * WAV_SECOND:
                    wav_std[:, :wavform.shape[1]] = wavform
                else:
                    wav_std = wavform[:, : sr * WAV_SECOND]
                wavform = wav_std

                wavform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wavform)
                output = self.model(wavform)
                # feature = output.last_hidden_state
                feature = output.extract_features
                feature = feature.detach().numpy().squeeze()
                feature = np.mean(feature, axis=0)
                person_feature.append(feature)
        feature = np.mean(np.array(person_feature), axis=0)

        # return torch.tensor(feature, dtype=torch.float, device=DEVICE), self.labels[item]
        return self.ids[item], feature, self.labels[item]


if __name__ == "__main__":
    dataset = AudioDataset()
    print(dataset[0])