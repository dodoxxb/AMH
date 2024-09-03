from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wavfile
import os
import json
import numpy as np

def get_voice_feature(wav_path, avg_voice_length):
    # print(wav_path)
    (rate, sig) = wavfile.read(wav_path)
    length = sig.shape[0]
    print(length)
    # if sig.shape[0]、atenate((sig, np.zeros(avg_voice_length-length)))

    # print(sig.shape)
    mfcc_feat = mfcc(sig, rate)
    logfbank_feat = logfbank(sig, rate)

    mfcc_feat = mfcc_feat.reshape(1, -1)
    # mfcc_feat = mfcc_feat.tolist()
    logfbank_feat = logfbank_feat.reshape(1, -1)

    # print("mfcc feature:",mfcc_feat.shape)
    # print("fbank feature:",logfbank_feat.shape)
    # print("MFCC:{mfcc}".format(mfcc=mfcc_feat))
    # print("LogFbank:{logfbank}".format(logfbank=logfbank_feat))

    return mfcc_feat.tolist(), logfbank_feat.tolist(), length

if __name__ == '__main__':
    MFCC = []
    LOGfbank = []

    v_length = []
    avg_voice_length = 121124
    max_voice_length = 508778

    root = os.path.dirname(os.getcwd())
    print("root direction", root)

    dataset_class = os.listdir(root + '/Dataset')
    print(dataset_class)

    for sample_class in dataset_class:
        sample_class_path = root + '/Dataset' + '/' + sample_class
        # print(sample_class_path)
        sample_file = os.listdir(sample_class_path)
        # print(sample_file)

        #样本内文件
        for detail in sample_file:
            detail_path = sample_class_path + '/' + detail
            sample_detail = os.listdir(detail_path)
            print(detail_path)
            temp_MFCC = []
            temp_LOGfbank = []

            for wav in sample_detail:
                if '.wav' in wav:
                    #找到文件夹中的录音文件

                    temp_1, temp_2, voice_len = get_voice_feature(detail_path + '/' + wav, max_voice_length)
                    temp_MFCC.append(temp_1)
                    # temp_LOGfbank.append(temp_2)
                    v_length.append(voice_len)
                    # print(np.array(MFCC).shape)
                else:
                    continue

            # avg_MFCC = np.mean(temp_MFCC, axis=0)
            # avg_LOGfbank = np.mean(temp_LOGfbank, axis=0)
            # temp_MFCC = np.array(temp_MFCC).reshape(1, -1)
            # temp_LOGfbank = np.array(temp_LOGfbank).reshape(1, -1)


            # temp_MFCC = temp_MFCC.tolist()
            # temp_LOGfbank = temp_LOGfbank.tolist()
            # MFCC.append(avg_MFCC)
            # LOGfbank.append(avg_LOGfbank)
            # print(np.array(MFCC).shape)
            # print(np.array(LOGfbank).shape)
            # print(temp_MFCC)
        # v_length = np.array(v_length).reshape(417, -1)
        print(np.array(v_length).shape) # 录音平均长度121124
        print(np.max(v_length))