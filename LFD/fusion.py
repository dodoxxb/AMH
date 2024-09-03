# coding = utf-8
"""
数据：
已经生成的feature，位置在feature/文件夹下
"""
# imports
import numpy as np
from data_util import load_data, shuffle_dataset
from uni_modal import train_best_cls
from sklearn import preprocessing


def late_fusion(text, wav, face):
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

    cls_text, _ = train_best_cls(text_x_train, text_y_train, text_x_test, text_y_test,
                                                                 l_text_data, l_text_label)
    # print("text modal, best test score:", best_test_score, " best large score: ", best_large_score)
    cls_wav, _ = train_best_cls(wav_x_train, wav_y_train, wav_x_test, wav_y_test,
                                                                l_wav_data, l_wav_label)
    # print("audio modal, best test score:", best_test_score, " best large score: ", best_large_score)
    cls_face, _ = train_best_cls(face_x_train, face_y_train, face_x_test, face_y_test,
                                                                l_face_data, l_face_label)
    # print("video modal, best test score:", best_test_score, " best large score: ", best_large_score)

    decision_x_train = np.vstack((cls_text.predict(text_x_train),
                                  cls_wav.predict(wav_x_train),
                                  cls_face.predict(face_x_train)))
    decision_x_train = decision_x_train.T
    decision_x_test = np.vstack((cls_text.predict(text_x_test),
                                 cls_wav.predict(wav_x_test),
                                 cls_face.predict(face_x_test)))
    decision_x_test = decision_x_test.T
    decision_x_l = np.vstack((cls_text.predict(l_text_data),
                              cls_wav.predict(l_wav_data),
                              cls_face.predict(l_face_data)))
    decision_x_l = decision_x_l.T
    decision_y_train = wav_y_train
    decision_y_test = wav_y_test
    decision_y_l = l_wav_label

    cls, scores = train_best_cls(decision_x_train, decision_y_train,
                                                            decision_x_test, decision_y_test,
                                                            decision_x_l, decision_y_l)
    return scores


def early_fusion(text, wav, face):
    text_data, text_label, l_text_data, l_text_label = text
    wav_data, wav_label, l_wav_data, l_wav_label = wav
    face_data, face_label, l_face_data, l_face_label = face

    x = np.concatenate((text_data, wav_data, face_data), axis=1)
    y = text_label
    x_large = np.concatenate((l_text_data, l_wav_data, l_face_data), axis=1)
    y_large = l_text_label

    index = shuffle_dataset(y.shape[0])
    x = x[index]
    y = y[index]

    num_train = int(np.round(text_label.shape[0] * 0.75))
    x_train = x[:num_train]
    x_test = x[num_train:]
    y_train = y[:num_train]
    y_test = y[num_train:]

    cls, best_scores = train_best_cls(x_train, y_train, x_test, y_test, x_large, y_large)
    # print("score on small test set:", best_test_score)
    # print("score on large data set:", best_large_score)
    return best_scores


if __name__ == "__main__":
    text = load_data("text")
    wav = load_data("wav")
    face = load_data("face")
    iteration = 500

    precision, recall, accuracy, f1 = [], [], [], []
    l_precision, l_recall, l_accuracy, l_f1 = [], [], [], []
    for iter in range(iteration):
        print("iteration: ", iter)
        scores = late_fusion(text, wav, face) # scores includes: precision/recall/acc/f1
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

