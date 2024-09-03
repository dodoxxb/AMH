"""
关于数据预处理的设置
主要包括数据地址
"""

ORIGINAL_DATA_PATH = {
    "data":r"../../data/data",
    "healthy": r"../../data/data/healthy",
    "unhealthy": r"../../data/data/unhealthy"
}

AUDIO_DENOISED_PATH = {
    "healthy": "../data_preprocessed/wav_healthy",
    "unhealthy": "../data_preprocessed/wav_unhealthy"
}

TEXT_CSV_PATH = {
    "healthy" : "../data_preprocessed/text_healthy.csv",
    "unhealthy": "../data_preprocessed/text_unhealthy.csv"
}

IMG_FACE_PATH = {
    "healthy" : "../data_preprocessed/img_healthy",
    "unhealthy" : "../data_preprocessed/img_unhealthy"
}