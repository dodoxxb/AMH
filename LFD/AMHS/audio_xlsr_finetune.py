
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

class AudioNet_ft(torch.nn.Module):
    def __intit__(self, model_choice, n_input, n_hidden, n_output):
        super(AudioNet_ft, self).__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_choice)
        self.hidden1 = torch.nn.Linear(n_input, n_hidden)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.wav2vec(x)


if __name__=="__main__":
    model_choice = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
    model = Wav2Vec2Model.from_pretrained(model_choice)
    wav_denoised_path = r"data_preprocessed\wav_healthy\00002-0101\00002-0101-wKgIb2CiGLSAWWhkAAQ9AHdR2yQ370_denoised.wav"
    second = 5
    wavform, sr = torchaudio.load(wav_denoised_path)
    wav_std = torch.zeros((1, sr * second))
    if wavform.shape[1] <= sr * second:  # 音频长度不足
        wav_std[:, :wavform.shape[1]] = wavform
    else:
        wav_std = wavform[:, : sr * second]
    wavform = wav_std
    wavform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wavform)
    output = model(wavform)
    for item in output:
        print(item)
    features = output["extract_features"]
    print(features.shape)

    hidden_state = output["last_hidden_state"]
    print(hidden_state.shape)