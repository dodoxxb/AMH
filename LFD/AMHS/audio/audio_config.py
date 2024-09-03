# coding = utf-8
# preprocess config
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CHOICE = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
WAV_SECOND = 5

AUDIO_RAW_PATH = {
    "healthy": "../../data/data/healthy",
    "unhealthy": "../../data/data/unhealthy"
}

AUDIO_DENOISED_PATH = {
    "healthy": "../data_preprocessed/wav_healthy",
    "unhealthy": "../data_preprocessed/wav_unhealthy",
}

AUDIO_FEATURE_PATH = {
    "healthy": "../data_preprocessed/wav_healthy_feature",
    "unhealthy": "../data_preprocessed/wav_unhealthy_feature"
}

TEXT_CSV_FILES = {
    "healthy": "../data_preprocessed/text_healthy.csv",
    "unhealthy": "../data_preprocessed/text_unhealthy.csv",
}
