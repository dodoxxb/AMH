# coding = utf-8

import numpy as np
import os
import random
from sklearn import preprocessing


feature_path = "feature"
filename = {
    "healthy_text":"text_healthy_feature.csv",
    "unhealthy_text":"text_unhealthy_feature.csv",
    "healthy_wav":"wav_healthy_feature.csv",
    "unhealthy_wav":"wav_unhealthy_feature.csv",
    "healthy_face":"face_healthy_feature.csv",
    "unhealthy_face":"face_unhealthy_feature.csv"
}


def load_data(modal="text"):

    if modal == "text":
        healthy = np.loadtxt(os.path.join(feature_path, filename["healthy_text"]), delimiter=",")
        unhealthy = np.loadtxt(os.path.join(feature_path, filename["unhealthy_text"]), delimiter=",")
    elif modal == "wav":
        healthy = np.loadtxt(os.path.join(feature_path, filename["healthy_wav"]), delimiter=",")
        unhealthy = np.loadtxt(os.path.join(feature_path, filename["unhealthy_wav"]), delimiter=",")
    else:
        healthy = np.loadtxt(os.path.join(feature_path, filename["healthy_face"]), delimiter=",")
        unhealthy = np.loadtxt(os.path.join(feature_path, filename["unhealthy_face"]), delimiter=",")

    num_neg = unhealthy.shape[0]
    num_pos = healthy.shape[0]
    data = np.concatenate((healthy[:num_neg], unhealthy))
    label = np.concatenate((np.zeros(num_neg), np.ones(num_neg)))
    large_data = np.concatenate((healthy, unhealthy))
    large_label = np.concatenate((np.zeros(num_pos), np.ones(num_neg)))

    # scale
    data = preprocessing.scale(data)
    large_data = preprocessing.scale(large_data)

    return data, label, large_data, large_label


def shuffle_dataset(total_num):
    index = [i for i in range(total_num)]
    random.shuffle(index)
    return index