# coding = utf-8

# imports
import numpy as np
from data_util import load_data, shuffle_dataset
from MKLpy.algorithms import AverageMKL, EasyMKL, GRAM, RMKL, MEMO, PWMK, FHeuristic, CKA
from MKLpy.metrics import pairwise
from MKLpy.preprocessing import normalization, rescale_01
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def train_MKL(text, wav, face):
    text_data, text_label, l_text_data, l_text_label = text
    wav_data, wav_label, l_wav_data, l_wav_label = wav
    face_data, face_label, l_face_data, l_face_label = face

    x = np.concatenate((text_data, wav_data, face_data), axis=1)
    y = text_label
    x_large = np.concatenate((l_text_data, l_wav_data, l_face_data), axis=1)
    y_large = l_text_label

    x = rescale_01(x)  # feature scaling in [0,1]
    x = normalization(x)  # ||X_i||_2^2 = 1

    index = shuffle_dataset(y.shape[0])
    x = x[index]
    y = y[index]

    num_train = int(np.round(text_label.shape[0] * 0.75))
    x_train = x[:num_train]
    x_test = x[num_train:]
    y_train = y[:num_train]
    y_test = y[num_train:]

    KLtr = [pairwise.homogeneous_polynomial_kernel(x_train, degree=d) for d in range(11)]
    KLte = [pairwise.homogeneous_polynomial_kernel(x_test, x_train, degree=d) for d in range(11)]
    l_KLte = [pairwise.homogeneous_polynomial_kernel(x_large, x_train, degree=d) for d in range(11)]
    # AverageMKL, EasyMKL, GRAM, RMKL, MEMO, PWMK, FHeuristic, CKA
    # clf = AverageMKL().fit(KLtr, y_train)  # a wrapper for averaging kernels
    # clf = EasyMKL(lam=0.1).fit(KLtr, y_train) # great
    # clf = GRAM().fit(KLtr, y_train)  # have warning, very slow, sometimes error, but excellent
    clf = RMKL().fit(KLtr, y_train) # great but slow
    # clf = MEMO().fit(KLtr, y_train) # not so good but ok
    # clf = PWMK().fit(KLtr, y_train) # great
    # clf = FHeuristic().fit(KLtr, y_train) # hopeful!
    # clf = CKA().fit(KLtr, y_train) # <0.5

    y_pred = clf.predict(KLte)  # predictions
    # y_score = clf.decision_function(KLte)  # rank
    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    y_pred = clf.predict(l_KLte)
    l_precision = precision_score(y_large, y_pred)
    l_accuracy = accuracy_score(y_large, y_pred)
    l_recall = recall_score(y_large, y_pred)
    l_f1 = f1_score(y_large, y_pred)

    return precision, recall, accuracy, f1, l_precision, l_recall, l_accuracy, l_f1


if __name__ == "__main__":
    # prepare data
    text = load_data("text")
    wav = load_data("wav")
    face = load_data("face")

    precision, recall, accuracy, f1 = [], [], [], []
    l_precision, l_recall, l_accuracy, l_f1 = [], [], [], []
    for i in range(100):
        print("iteration:", i)
        scores = train_MKL(text, wav, face)
        print(scores)
        precision.append(scores[0])
        recall.append(scores[1])
        accuracy.append(scores[2])
        f1.append(scores[3])
        l_precision.append(scores[4])
        l_recall.append(scores[5])
        l_accuracy.append(scores[6])
        l_f1.append(scores[7])
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