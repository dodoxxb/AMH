# imports
import os

import cv2
import cv2 as cv
import numpy as np
from preproc_config import ORIGINAL_DATA_PATH, IMG_FACE_PATH


def exist_face(img, haar_cascade):
    """
    检测图像img中是否有人脸
    :param img:
    :return:
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=3) # 使用默认的参数
    if not len(faces_rect) == 0:
        return True
    else:
        return False

def extract_face(path, processed_path):
    """
        遍历每个文件夹，
        找出能够识别出人脸的图像
        存入data_preprocessed文件夹中对应的位置
    """
    haar_cascade = cv.CascadeClassifier("haar_face.xml")

    for root, folders, files in os.walk(path):
        if len(folders) == 0:
            for item in files:
                if ".png" in item:
                    img_path = os.path.join(root, item)
                    img = cv.imread(img_path)
                    if exist_face(img, haar_cascade):
                        """
                        识别为人脸。把png文件移到preprocessed_path文件夹下对应的地方
                        这里的效果不是很好所以我又人工筛了一遍，去除了一些明显不符合的图像
                        """
                        new_path = os.path.join(processed_path, root[root.rfind("\\")+1:])
                        if not os.path.exists(new_path):
                            os.mkdir(new_path)
                        img = cv.imread(img_path)
                        cv.imwrite(os.path.join(new_path, item), img)
    print("finish!")


def normalize_face(path, processed_path):
    """
    便历文件夹中每个图象， 对人脸图像进行归一化，目的是调整亮度对比度
    存在目标文件夹下
    :param path: 要修改的图像的路径（文件夹）
    :param processed_path: 修改过后的存储路径（文件夹）
    :return: 没有
    """
    for root, folders, files in os.walk(path):
        if len(folders) == 0:
            for item in files:
                img_path = os.path.join(root, item)
                print(img_path)
                img = cv.imread(img_path)
                # img = cv.cvtColor(img, cv2.COLOR_RGB2GRAY)
                dst = np.zeros(img.shape)
                img = cv.normalize(img, dst, 0, 255, cv2.NORM_MINMAX)
                new_path = os.path.join(processed_path, img_path[-14:])
                print(new_path)
                cv.imwrite(os.path.join(new_path), img)
    print("finish!")


if __name__ == "__main__":
    normalize_face(r"F:\white_fish\projects\fang\AMHS\data_preprocessed\single_img\healthy",
                   r"F:\white_fish\projects\fang\AMHS\data_preprocessed\single_img\healthy_normalized")
    normalize_face(r"F:\white_fish\projects\fang\AMHS\data_preprocessed\single_img\unhealthy",
                   r"F:\white_fish\projects\fang\AMHS\data_preprocessed\single_img\unhealthy_normalized")