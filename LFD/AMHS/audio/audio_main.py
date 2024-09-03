# coding = utf-8

# imports
import random
import os
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from audio_data_util import AudioDataset

feature_path = "."
filename = {
    "healthy_wav": "wav_healthy_feature.csv",
    "unhealthy_wav": "wav_unhealthy_feature.csv",
}


def load_data():
    healthy = np.loadtxt(os.path.join(feature_path, filename["healthy_wav"]), delimiter=",")
    unhealthy = np.loadtxt(os.path.join(feature_path, filename["unhealthy_wav"]), delimiter=",")

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


if __name__ == "__main__":
    """
    this part would be quite slow due to calculating the feature of wav files is time-consuming
    strongly recommend to store the result of feature if there is no fine-tuning.  
    """
    # dataset = AudioDataset(train=True)
    # x = []
    # y = []
    # for _, feature, label in dataset:
    #     x.append(feature)
    #     y.append(label)
    #
    # print("data_loaded")

    """
    the feature of the wav files are stored in AUDIO_FEATURE_PATH
    """
    # x = np.array(x)
    # y = np.array(y)
    # leng = len(dataset)
    data = load_data()
    x, y, l_x, l_y = data
    index = [i for i in range(y.shape[0])]
    random.shuffle(index)
    x = x[index]
    y = y[index]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    cls = SVC()  # DNN might even be worse
    cls.fit(x_train, y_train)
    pre = cls.predict(x_test)

    print(accuracy_score(pre, y_test))   # my result: in 10 trials, average acc is 0.56, the max acc is 0.64