"""
fucking deep learning
"""
import librosa
import soundfile
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import os
import pandas as pd
import random


def extract_feature(filename, feature_type):
    with soundfile.SoundFile(filename) as sf:
        X = sf.read(dtype = "float32")
        sr = sf.samplerate

        result = {"mfcc": None, "chroma": None, "mel": None, "contrast": None, "tonnetz": None}
        if "mfcc" in feature_type:
            mfccs = librosa.feature.mfcc(y=X, sr=sr) # 结果是a*b。增加参数n_mfcc并且改变a的值，b由音频长度决定
            mfccs = np.mean(mfccs, axis=1)
            result["mfcc"] = mfccs
        if "chroma" in feature_type:
            stft = np.abs(librosa.stft(X))
            chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
            chroma = np.mean(chroma, axis=1)
            result["chroma"] = chroma
        if "mel" in feature_type:
            mel = librosa.feature.melspectrogram(X, sr)
            mel = np.mean(mel, axis=1)
            result["mel"] = mel
        if "contrast" in feature_type:
            stft = np.abs(librosa.stft(X))
            contrast = librosa.feature.spectral_contrast(X, sr)
            contrast = np.mean(contrast, axis=1)
            result["contrast"] = contrast
        if "tonnetz" in feature_type:
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr)
            tonnetz = np.mean(tonnetz, axis=1)
            result["tonnetz"] = tonnetz
    return result


def generate_dataset(root_dir, feature_type, in_train = True):
    unhealthy_csv = pd.read_csv(os.path.join(root_dir, "unhealthy_samples.csv"))
    unhealthy_ids = list(unhealthy_csv["id"])
    unhealthy_labels = [0] * len(unhealthy_ids)
    num = len(unhealthy_ids)
    healthy_csv = pd.read_csv(os.path.join(root_dir, "healthy_samples.csv"))
    healthy_ids = list(healthy_csv["id"])
    healthy_labels = [1] * len(healthy_ids)
    if in_train:
        healthy_ids = healthy_ids[: num]
        healthy_labels = healthy_labels[: num]

    ids = healthy_ids + unhealthy_ids
    labels = healthy_labels + unhealthy_labels

    dataset = {"mfcc": [], "chroma": [], "mel": [], "contrast": [], "tonnetz": []}
    for i in range(len(ids)):
        unique_id = ids[i]
        label = labels[i]
        data_path = os.path.join(root_dir + "/wav_healthy", unique_id)
        if label == 0:  # unhealthy
            data_path = os.path.join(root_dir + "/wav_unhealthy", unique_id)
        features = {"mfcc": [], "chroma": [], "mel": [], "contrast": [], "tonnetz": []}

        print("processing "+data_path)

        for file in os.listdir(data_path):
            filename = os.path.join(data_path, file)
            result = extract_feature(filename, feature_type)
            features["mfcc"].append(result["mfcc"])
            features["chroma"].append(result["chroma"])
            features["mel"].append(result["mel"])
            features["contrast"].append(result["contrast"])
            features["tonnetz"].append(result["tonnetz"])

        features["mfcc"] = np.array(features["mfcc"])
        features["chroma"] = np.array(features["chroma"])
        features["mel"] = np.array(features["mel"])
        features["contrast"] = np.array(features["contrast"])
        features["tonnetz"] = np.array(features["tonnetz"])

        dataset["mfcc"].append(np.concatenate(features["mfcc"]))
        dataset["chroma"].append(np.concatenate(features["chroma"]))
        dataset["mel"].append(np.concatenate(features["mel"]))
        dataset["contrast"].append(np.concatenate(features["contrast"]))
        dataset["tonnetz"].append(np.concatenate(features["tonnetz"]))

    dataset["mfcc"] = np.array(dataset["mfcc"])
    dataset["chroma"] = np.array(dataset["chroma"])
    dataset["mel"] = np.array(dataset["mel"])
    dataset["contrast"] = np.array(dataset["contrast"])
    dataset["tonnetz"] = np.array(dataset["tonnetz"])
    return dataset, np.array(labels)


def generate_classifier():
    classifiers = []
    classifier_info = []

    # SVC ============================================
    classifiers.append(SVC(kernel="rbf"))
    classifier_info.append("svc_kernel_rbf")
    classifiers.append(SVC(kernel="poly"))
    classifier_info.append("svc_kernel_poly")
    classifiers.append(SVC(kernel="sigmoid"))
    classifier_info.append("svc_kernel_sigmoid")

    # RandomFroest ============================================
    classifiers.append(RandomForestClassifier(n_estimators=100, criterion="gini"))
    classifier_info.append("randomforest_estimators_100_gini")
    classifiers.append(RandomForestClassifier(n_estimators=100, criterion="entropy"))
    classifier_info.append("randomforest_estimators_100_entropy")
    classifiers.append(RandomForestClassifier(n_estimators=200, criterion="gini"))
    classifier_info.append("randomforest_estimators_200_gini")
    classifiers.append(RandomForestClassifier(n_estimators=200, criterion="entropy"))
    classifier_info.append("randomforest_estimators_200_entropy")

    # GradientBoosting ============================================
    classifiers.append(GradientBoostingClassifier(loss="deviance"))
    classifier_info.append("gradientboosting_loss_deviance")
    classifiers.append(GradientBoostingClassifier(loss="exponential"))
    classifier_info.append("gradientboosting_loss_exponential")

    # KNN ============================================
    classifiers.append(KNeighborsClassifier(n_neighbors=5, weights="uniform"))
    classifier_info.append("kneighbor_n_5_weights_uniform")
    classifiers.append(KNeighborsClassifier(n_neighbors=3, weights="uniform"))
    classifier_info.append("kneighbor_n_7_weights_uniform")
    classifiers.append(KNeighborsClassifier(n_neighbors=5, weights="distance"))
    classifier_info.append("kneighbor_n_5_weights_distance")
    classifiers.append(KNeighborsClassifier(n_neighbors=3, weights="distance"))
    classifier_info.append("kneighbor_n_7_weights_distance")

    # Bagging ============================================
    classifiers.append(BaggingClassifier(SVC()))
    classifier_info.append("bagging_svc")
    classifiers.append(BaggingClassifier(RandomForestClassifier()))
    classifier_info.append("bagging_randomforest")
    classifiers.append(BaggingClassifier(KNeighborsClassifier()))
    classifier_info.append("bagging_kneighbor")

    # Adaboost==============================================
    classifiers.append(AdaBoostClassifier(SVC(),algorithm='SAMME'))
    classifier_info.append("adaboost_svc")
    classifiers.append(AdaBoostClassifier(RandomForestClassifier(),algorithm='SAMME'))
    classifier_info.append("adaboost_randomforest")
    # MLP ============================================
    classifiers.append(MLPClassifier(hidden_layer_sizes=128,
                                     activation="relu",
                                     solver="adam",
                                     learning_rate="adaptive",
                                     max_iter=500))
    classifier_info.append("mlp_128_relu_adam_adaptive_500")
    classifiers.append(MLPClassifier(hidden_layer_sizes=128,
                                     activation="relu",
                                     solver="sgd",
                                     learning_rate="adaptive",
                                     max_iter=500))
    classifier_info.append("mlp_128_relu_sgd_adaptive_500")
    classifiers.append(MLPClassifier(hidden_layer_sizes=64,
                                     activation="relu",
                                     solver="adam",
                                     learning_rate="adaptive",
                                     max_iter=500))
    classifier_info.append("mlp_64_relu_adam_adaptive_500")
    classifiers.append(MLPClassifier(hidden_layer_sizes=64,
                                     activation="relu",
                                     solver="sgd",
                                     learning_rate="adaptive",
                                     max_iter=500))
    classifier_info.append("mlp_64_relu_sgd_adaptive_500")
    classifiers.append(MLPClassifier(hidden_layer_sizes=128,
                                     activation="relu",
                                     solver="adam",
                                     learning_rate="adaptive",
                                     max_iter=1000))
    classifier_info.append("mlp_128_relu_adam_adaptive_1000")
    classifiers.append(MLPClassifier(hidden_layer_sizes=128,
                                     activation="relu",
                                     solver="sgd",
                                     learning_rate="adaptive",
                                     max_iter=1000))
    classifier_info.append("mlp_128_relu_sgd_adaptive_1000")
    classifiers.append(MLPClassifier(hidden_layer_sizes=64,
                                     activation="relu",
                                     solver="adam",
                                     learning_rate="adaptive",
                                     max_iter=1000))
    classifier_info.append("mlp_64_relu_adam_adaptive_1000")
    classifiers.append(MLPClassifier(hidden_layer_sizes=64,
                                     activation="relu",
                                     solver="sgd",
                                     learning_rate="adaptive",
                                     max_iter=1000))
    classifier_info.append("mlp_64_relu_sgd_adaptive_1000")

    return classifiers, classifier_info


def choose_best_classifier(x_train, y_train, x_test, y_test):
    classifiers, classifier_info = generate_classifier()
    best_classifier = None
    best_classifier_info = None
    best_acc = 0.0
    for i in range(len(classifiers)):
        model = classifiers[i]
        model.fit(x_train, y_train)
        result = model.predict(x_test)
        acc_score = accuracy_score(result, y_test)
        print("classifier: ", classifier_info[i], ", test acc: %.4f" % (acc_score))
        if acc_score > best_acc:
            best_acc = acc_score
            best_classifier = model
            best_classifier_info = classifier_info[i]

    return best_classifier, best_classifier_info, best_acc


def find_best_auto(names, labels, mfcc, chroma, mel, contrast, tonnetz):
    # 分训练集测试集
    index = [i for i in range(labels.shape[0])]
    random.shuffle(index)
    mfcc = mfcc[index]
    mel = mel[index]
    chroma = chroma[index]
    contrast = contrast[index]
    tonnetz = tonnetz[index]
    labels = labels[index]

    # 前80%做训练数据
    train_size = 0.8
    sample_num = labels.shape[0]
    train_num = int(train_size * sample_num)

    mfcc_train = mfcc[:train_num]
    chroma_train = chroma[:train_num]
    mel_train = mel[:train_num]
    contrast_train = contrast[:train_num]
    tonnetz_train = tonnetz[:train_num]
    mfcc_test = mfcc[train_num:]
    chroma_test = chroma[train_num:]
    mel_test = mel[train_num:]
    contrast_test = contrast[train_num:]
    tonnetz_test = tonnetz[train_num:]
    labels_train = labels[:train_num]
    labels_test = labels[train_num:]

    x_train = [mfcc_train, chroma_train, mel_train, contrast_train, tonnetz_train]
    x_test = [mfcc_test, chroma_test, mel_test, contrast_test, tonnetz_test]
    y_train = labels_train
    y_test = labels_test
    best_models = []
    best_info = []
    best_accs = []
    for i in range(5):
        print(">>>choosing best classifier for: ", names[i], "...")
        cls, cls_info, best_acc = choose_best_classifier(x_train[i], y_train, x_test[i], y_test)
        print("The best classifier for ", names[i], " is ", cls_info, " with acc: ", best_acc)
        best_models.append(cls)
        best_info.append(cls_info)
        best_accs.append(best_acc)

    print("=" * 20)
    samples = [mfcc, chroma, mel, contrast, tonnetz]
    decisions = [best_models[i].predict(samples[i]) for i in range(len(names))]
    decisions = np.array(decisions)
    decisions = decisions.T

    decision_train = decisions[:train_num]
    decision_test = decisions[train_num:]
    cls, cls_info, best_acc = choose_best_classifier(decision_train, labels_train, decision_test, labels_test)
    print(best_info)
    print(best_accs)
    print(cls_info)
    print(best_acc)
    best_models.append(cls)
    best_info.append(cls_info)
    best_dataset = index

    return best_models, best_info, best_dataset, best_acc

if __name__ == "__main__":
    # dataset, labels = generate_dataset("./data_preprocessed", ["mfcc", "chroma", "mel", "contrast", "tonnetz"], in_train= False)
    # np.save("mfcc_large.npy", dataset["mfcc"])
    # np.save("chroma_large.npy", dataset["chroma"])
    # np.save("mel_large.npy", dataset["mel"])
    # np.save("contrast_large.npy", dataset["contrast"])
    # np.save("tonnetz_large.npy", dataset["tonnetz"])
    # np.save("labels_large.npy", labels)
    # print("finish!")
    mfcc = np.load("mfcc.npy") # 190, 100
    chroma = np.load("chroma.npy") # 190, 60
    mel = np.load("mel.npy") # 190, 640
    contrast = np.load("contrast.npy") # 190, 35
    tonnetz = np.load("tonnetz.npy") # 190, 30
    labels = np.load("labels.npy") # 190,
    mfcc_large = np.load("mfcc_large.npy")
    chroma_large = np.load("chroma_large.npy")
    mel_large = np.load("mel_large.npy")
    contrast_large = np.load("contrast_large.npy")
    tonnetz_large = np.load("tonnetz_large.npy")
    labels_large = np.load("labels_large.npy")

    # 归一化
    mfcc = mfcc/ np.max(mfcc)
    chroma = chroma/ np.max(chroma)
    mel = mel/ np.max(mel)
    contrast = contrast/ np.max(contrast)
    tonnetz = tonnetz/ np.max(tonnetz)
    mfcc_large = mfcc_large / np.max(mfcc_large)
    chroma_large = chroma_large / np.max(chroma_large)
    mel_large = mel_large / np.max(mel_large)
    contrast_large = contrast_large / np.max(contrast_large)
    tonnetz_large = tonnetz_large / np.max(tonnetz_large)
    test_large = [mfcc_large, chroma_large, mel_large, contrast_large, tonnetz_large]

    # find best result in 100 iters
    iterations = 100
    best_models = None # [cls for mfcc, cls for chroma, ..., cls for tonnetz, final cls]
    best_dataset = None # index of samples in the training set
    best_info = None
    best_acc = 0.0
    best_acc_large = 0.0
    for t in range(iterations):
        print(">>>the ", t, "th iteration starts...")
        try:
            models, models_info, dataset, acc = find_best_auto(["mfcc", "chroma", "mel", "contrast", "tonnetz"], labels, mfcc, chroma, mel, contrast, tonnetz )
        except Exception as e:
            print(e)
            continue
        decisions_large = np.array([models[i].predict(test_large[i]) for i in range(len(test_large))]).T
        result_large = models[-1].predict(decisions_large)
        large_acc = accuracy_score(result_large, labels_large)
        print("result on large dataset is: ", large_acc)
        if large_acc > best_acc_large:
            best_acc_large = large_acc
            best_acc = acc
            best_info = models_info
            best_models = models
            best_dataset = dataset

    np.save("best_audio_split.npy",np.array(best_dataset))
    print("="*30)
    print("With models \n", best_info[:-1], " for classification of mfcc, chroma, mel, contrast, tonnetz features \n and ",\
          best_info[-1], " for final classification\nThe best acc on large test set is: ",best_acc_large)

    # test
    # 生成获得最好结果时候的训练集
    # index = np.load("best_audio_split.npy")
    # mfcc_train = mfcc[index]
    # chroma_train = chroma[index]
    # mel_train = mel[index]
    # contrast_train = contrast[index]
    # tonnetz_train = tonnetz[index]
    # labels_train = labels[index]
    #
    # # 用这个训练集训练以上结果中的分类器
    # cls_mfcc = RandomForestClassifier(n_estimators=100, criterion="gini")
    # cls_chroma = MLPClassifier(hidden_layer_sizes=64,
    #                                  activation="relu",
    #                                  solver="adam",
    #                                  learning_rate="adaptive",
    #                                  max_iter=500)
    # cls_mel = KNeighborsClassifier(n_neighbors=3, weights="uniform")
    # cls_contrast = RandomForestClassifier(n_estimators=200, criterion="gini")
    # cls_tonnetz = KNeighborsClassifier(n_neighbors=3, weights="uniform")
    # cls_final = MLPClassifier(hidden_layer_sizes=64,
    #                                  activation="relu",
    #                                  solver="sgd",
    #                                  learning_rate="adaptive",
    #                                  max_iter=500)
    #
    # cls_mfcc.fit(mfcc_train, labels_train)
    # cls_chroma.fit(chroma_train,labels_train)
    # cls_mel.fit(mel_train,labels_train)
    # cls_contrast.fit(contrast_train,labels_train)
    # cls_tonnetz.fit(tonnetz_train,labels_train)
    # p_mfcc = cls_mfcc.predict(mfcc_train)
    # p_chroma = cls_chroma.predict(chroma_train)
    # p_mel = cls_mel.predict(mel_train)
    # p_contrast = cls_contrast.predict(contrast_train)
    # p_tonnetz = cls_tonnetz.predict(tonnetz_train)
    # print("mfcc train acc: %.4f" % (accuracy_score(p_mfcc, labels_train)))
    # print("chroma train acc: %.4f" % (accuracy_score(p_chroma, labels_train)))
    # print("mel train acc: %.4f" % (accuracy_score(p_mel, labels_train)))
    # print("contrast train acc: %.4f" % (accuracy_score(p_contrast, labels_train)))
    # print("tonnetz train acc: %.4f" % (accuracy_score(p_tonnetz, labels_train)))
    #
    # p_mfcc_ = cls_mfcc.predict(mfcc)
    # p_chroma_ = cls_chroma.predict(chroma)
    # p_mel_ = cls_mel.predict(mel)
    # p_contrast_ = cls_contrast.predict(contrast)
    # p_tonnetz_ = cls_tonnetz.predict(tonnetz)
    # print("mfcc small dataset acc: %.4f" % (accuracy_score(p_mfcc_, labels)))
    # print("chroma small dataset acc: %.4f" % (accuracy_score(p_chroma_, labels)))
    # print("mel small dataset acc: %.4f" % (accuracy_score(p_mel_, labels)))
    # print("contrast small dataset acc: %.4f" % (accuracy_score(p_contrast_, labels)))
    # print("tonnetz small dataset acc: %.4f" % (accuracy_score(p_tonnetz_, labels)))
    #
    # final_train = np.array([p_mfcc, p_chroma, p_mel, p_contrast, p_tonnetz]).T
    # cls_final.fit(final_train, labels_train)
    # p_final = cls_final.predict(final_train)
    # print("final train acc: %.4f" % (accuracy_score(p_final, labels_train)))
    # print(p_final)
    #
    # final = np.array([p_mfcc_, p_chroma_, p_mel_, p_contrast_, p_tonnetz_]).T
    # p_final_ = cls_final.predict(final)
    # print("final small dataset acc: %.4f" % (accuracy_score(p_final_, labels)))
    #
    # mfcc_large = np.load("mfcc_large.npy")
    # chroma_large = np.load("chroma_large.npy")
    # mel_large = np.load("mel_large.npy")
    # contrast_large = np.load("contrast_large.npy")
    # tonnetz_large = np.load("tonnetz_large.npy")
    # labels_large = np.load("labels_large.npy")
    #
    # mfcc_large = mfcc_large / np.max(mfcc_large)
    # chroma_large = chroma_large / np.max(chroma_large)
    # mel_large = mel_large / np.max(mel_large)
    # contrast_large = contrast_large / np.max(contrast_large)
    # tonnetz_large = tonnetz_large / np.max(tonnetz_large)
    #
    # p_mfcc = cls_mfcc.predict(mfcc_large)
    # p_chroma = cls_chroma.predict(chroma_large)
    # p_mel = cls_mel.predict(mel_large)
    # p_contrast = cls_contrast.predict(contrast_large)
    # p_tonnetz = cls_tonnetz.predict(tonnetz_large)
    # final = np.array([p_mfcc, p_chroma, p_mel, p_contrast, p_tonnetz]).T
    # p_final = cls_final.predict(final)
    # print("large dataset acc: %.4f"%(accuracy_score(p_final, labels_large)))
