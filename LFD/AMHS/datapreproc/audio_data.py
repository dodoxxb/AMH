"""
这个文件的作用：
把录音转成specgram然后存到对应的文件夹
对音频数据降噪, 降噪后的数据加上denoised标签，保存到data_preprocessed文件夹下
把降噪后的录音转成specgram存到对应的文件夹
mfcc特征
"""
import os
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import librosa
import librosa.display
import noisereduce as nr
import numpy as np
import cv2 as cv
import torchaudio
# from preproc_config import ORIGINAL_DATA_PATH, AUDIO_DENOISED_PATH

# get and draw mfcc feature
def get_mfcc(file_path = r"D:\projects\fang\data\data\unhealthy\00002-0110\wKgIb2CiGuGAcO8BAAQkAB7HpHY079.wav"):
    x, sr = librosa.load(file_path, duration=4, offset=0.5)
    mfccs = librosa.feature.mfcc(x, sr)
    print(mfccs.shape) # duration = 3: 20 130; duration = 4: 20 173;
    # Displaying  the MFCCs:
    plt.figure(figsize=(15, 7))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.show()

# draw the specgram
def get_specgram(data, fs, savepath = None, filename = None):
    plt.figure()
    plt.specgram(data, NFFT=256, Fs=fs)
    if savepath is not None:
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        plt.savefig(os.path.join(savepath, filename))
    plt.close()

def audio_to_specgram(path, processed_path):
    """
    检查每个文件夹下有多少wav文件。
    正常的话应该是5个。
        有问题的，输出文件夹名称和其中的wav数
        没有问题的，记录每个问题对应wav文件的长度
    :param path:
    :return:
    """
    for root, _, files in os.walk(path):
        if not len(files) == 0:
            for filename in files:
                if ".wav" in filename:
                    wav_path = os.path.join(root, filename)
                    fs, data = wav.read(wav_path)
                    this_id = root[root.rfind("\\")+1:]
                    new_file_path = os.path.join(processed_path, this_id)
                    new_file_name = this_id + "-" + filename[:-4] + ".png"
                    get_specgram(data, fs, savepath=new_file_path, filename = new_file_name)
    print("finish!")

def denoise(data, fs, savepath, filename):
    reduced_noise = nr.reduce_noise(y=data, sr=fs)
    wav.write(os.path.join(savepath, filename), rate=fs, data=reduced_noise)

def audio_denoise(path, processed_path):
    """
    检查每个文件夹下有多少wav文件。
    正常的话应该是5个。
        有问题的，输出文件夹名称和其中的wav数
        没有问题的，记录每个问题对应wav文件的长度
    :param path:
    :return:
    """
    for root, _, files in os.walk(path):
        if not len(files) == 0:
            for filename in files:
                if ".wav" in filename:
                    wav_path = os.path.join(root, filename)
                    fs, data = wav.read(wav_path)
                    this_id = root[root.rfind("\\")+1:]
                    new_file_path = os.path.join(processed_path, this_id)
                    new_file_name = this_id + "-" + filename[:-4] + "_denoised.wav"
                    denoise(data, fs, savepath=new_file_path, filename = new_file_name)
    print("finish!")

def denoised_audio_to_specgram(denoised_path, spec_path):
    for root, _, files in os.walk(denoised_path):
        if not len(files) == 0:
            count = 0
            for filename in files:
                if ".wav" in filename:
                    wav_path = os.path.join(root, filename)
                    fs, data = wav.read(wav_path)
                    this_id = root[root.rfind("\\")+1:]
                    new_file_path = os.path.join(spec_path, this_id)
                    new_file_name = this_id + "-%d.png"%count
                    get_specgram(data, fs, savepath=new_file_path, filename = new_file_name)
                    count += 1
            print("finish "+root)
    print("finish!")

def get_melspecgram(samples, sample_rate, savepath = None, filename = None):
    mel_sgram = librosa.feature.melspectrogram(samples, sr=sample_rate)
    mel_sgram = librosa.power_to_db(mel_sgram, ref=np.max)
    plt.figure()
    librosa.display.specshow(mel_sgram, sr=sample_rate)
    if savepath is not None:
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        plt.savefig(os.path.join(savepath, filename))
    plt.close()


def denoised_audio_to_melspecgram(denoised_path, melspec_path):
    for root, _, files in os.walk(denoised_path):
        if not len(files) == 0:
            count = 0
            for filename in files:
                if ".wav" in filename:
                    wav_path = os.path.join(root, filename)
                    # fs, data = wav.read(wav_path)
                    samples, sample_rate = librosa.load(wav_path, sr=None, duration=5)
                    #============================
                    second = 5
                    wav_std = np.zeros(sample_rate * second)
                    if samples.shape[0] <= sample_rate * second:  # 音频长度不足
                        wav_std[:samples.shape[0]] = samples
                    else:
                        wav_std = samples[: sample_rate * second]
                    samples = wav_std
                    #=============================
                    this_id = root[root.rfind("\\")+1:]
                    new_file_path = os.path.join(melspec_path, this_id)
                    new_file_name = this_id + "-%d.png"%count
                    get_melspecgram(samples, sample_rate, savepath=new_file_path, filename = new_file_name)
                    count += 1
            print("finish "+root)
    print("finish!")


if __name__ == "__main__":
    # audio_denoise(ORIGINAL_DATA_PATH["healthy"], AUDIO_DENOISED_PATH["healthy"])
    # audio_denoise(ORIGINAL_DATA_PATH["unhealthy"], AUDIO_DENOISED_PATH["unhealthy"])
    denoised_audio_to_melspecgram(r"D:\projects\fang\AMHS\data_preprocessed\wav_healthy",r"D:\projects\fang\AMHS\data_preprocessed\wav_melspec_healthy")
    denoised_audio_to_melspecgram(r"D:\projects\fang\AMHS\data_preprocessed\wav_unhealthy",r"D:\projects\fang\AMHS\data_preprocessed\wav_melspec_unhealthy")
    # AUDIO_FILE = r"D:\projects\fang\data\data\healthy\00002-0101\wKgIb2CiGLSAWWhkAAQ9AHdR2yQ370.wav"
    # samples, sample_rate = librosa.load(AUDIO_FILE, sr=None, offset=0.5, duration=5)
    # # sgram = librosa.stft(samples)
    # # sgram_mag, _ = librosa.magphase(sgram)
    # # framelength = 0.025
    # # framesize = int(framelength*sample_rate)
    # mel_sgram = librosa.feature.melspectrogram(samples, sr=sample_rate)
    # print(mel_sgram.shape)
    # mel_sgram = librosa.power_to_db(mel_sgram, ref=np.max)
    # print(mel_sgram.shape)
    # librosa.display.specshow(mel_sgram, sr=sample_rate)
    # plt.show()
    print("finish!")


