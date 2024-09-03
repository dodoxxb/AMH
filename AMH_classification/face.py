import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

if __name__ == '__main__':
    test = cv.imread("dataset/healthy/00002-0101/2021-05-17-15-18-19.png")
    test_rgb = cv.cvtColor(test, cv.COLOR_BGR2RGB)
    test_gray = cv.cvtColor(test_rgb, cv.COLOR_RGB2GRAY)

    plt.imshow(test_gray)

    # save facial landmark detection model's name as LBFmodel
    LBFmodel = "lbfmodel.yaml"

    # save face detection algorithm's name as haarcascade
    haarcascade = "haarcascade_frontalface_alt2.xml"

    # create an instance of the Face Detection Cascade Classifier
    detector = cv.CascadeClassifier(haarcascade)

    # Detect faces using the haarcascade classifier on the "grayscale image"
    faces = detector.detectMultiScale(test_gray)

    # create an instance of the Facial landmark Detector with the model
    landmark_detector = cv.face.createFacemarkLBF()
    landmark_detector.loadModel(LBFmodel)

    _, landmarks = landmark_detector.fit(test_gray, faces)

    landmarks = list(landmarks)

    landmarks = np.array(landmarks).reshape(-1,1)

    print(landmarks)