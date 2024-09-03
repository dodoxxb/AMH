1.数据预处理：

所有经过与处理的数据的保存路径都在./data_preprocessed文件夹下。
所有用于预处理的脚本路径都在./datapreproc文件夹下

文本数据：
预处理程序：./datapreproc/text_data.py
从数据集的json文件中整理出【回答内容的文本数据（5个）+选择题答案（3个）+回答问题时的录音文件名（5个）】，放在data_preprocessed文件夹下，文件名为text_healthy.csv和text_unhealthy.csv

音频数据：
预处理程序：./datapreproc/auido_data.py
经过降噪的音频数据，放在data_preprocessed/wav_healthy和data_preprocessed/wav_unhealthy文件夹下

图像数据：
预处理程序：./datapreproc/video_data.py
重新筛选人脸，去除不包含人脸的图像。将包含人脸的图像存放在img_healthy和img_unhealthy文件夹下。

2. 单模态特征提取和分类器训练
*fine-tune
文本：main_text.py
音频：main_audio_xlsr.py