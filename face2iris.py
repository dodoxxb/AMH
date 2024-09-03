import os
import mediapipe as mp
import cv2 as cv
import pandas as pd
import numpy as np

def get_iris(img_path):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    # 0代表近景人脸，1代表远景
    with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection:

        image = cv.imread(img_path)
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        if not results.detections:
            print("No iris found")
            return 0, 0, 0, 0


        for index, detection in enumerate(results.detections):
            # print(index, detection)
            # print(detection.location_data.relative_bounding_box)
            left_x = detection.location_data.relative_keypoints[0].x
            left_y = detection.location_data.relative_keypoints[0].y
            right_x = detection.location_data.relative_keypoints[1].x
            right_y = detection.location_data.relative_keypoints[1].y
            # print(detection.location_data.relative_keypoints[0].x, detection.location_data.relative_keypoints[0])

        # print(left_x, left_y, right_x, right_y)
        mp_drawing.draw_detection(image, detection)

        annotated_image = image.copy()
        for detection in results.detections:
            # print('Nose tip:')
            # print(mp_face_detection.get_key_point(
            #     detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
            mp_drawing.draw_detection(annotated_image, detection)
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, ic = image.shape
        bbox = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                int(bboxC.width * iw), int(bboxC.height * ih))
        # print(bboxC)
        cv.rectangle(image, bbox, (255, 0, 0), 5)  # 自定义绘制函数，不适用官方的mpDraw.draw_detection
        # cv.imshow('image', image)

        cv.waitKey(0)

    return left_x, left_y, right_x, right_y

def judge_face(img_path):
    judge = 1
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    # 0代表近景人脸，1代表远景
    with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection:

        image = cv.imread(img_path)
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        if not results.detections:
            judge = 0
        # print(results.detections)
    return judge

if __name__ == '__main__':
    dataset_class = os.listdir('./Dataset')
    print(dataset_class)
    leftx = []
    lefty = []
    rightx = []
    righty = []
    iris_length = []
    name = ['left_x', 'left_y', 'right_x', 'right_y']

    for sample_class in dataset_class:
        sample_class_path = './Dataset' + '/' + sample_class
        # print(sample_class_path)
        sample_file = os.listdir(sample_class_path)
        # print(sample_file)

        for detail in sample_file:
            detail_path = sample_class_path + '/' + detail
            sample_detail = os.listdir(detail_path)
            print(detail_path)

            for img in sample_detail:
                if 'png' in img and '2021' in img:
                    judge = judge_face(detail_path + '/' + img)
                    if judge == 1:
                        left_eye_x, left_eye_y, right_eye_x, right_eye_y = get_iris(detail_path + '/' + img)
                        leftx.append(left_eye_x)
                        lefty.append(left_eye_y)
                        rightx.append(right_eye_x)
                        righty.append(right_eye_y)
                        # print(left_eye_x, left_eye_y, right_eye_x, right_eye_y)
                    else:
                        continue

            for i in range(0, 684 - len(leftx)):
                leftx.append(np.mean(leftx))
                lefty.append(np.mean(lefty))
                rightx.append(np.mean(rightx))
                righty.append(np.mean(righty))
                # leftx.append(0)
                # lefty.append(0)
                # rightx.append(0)
                # righty.append(0)


            leftx = np.array(leftx).reshape(-1, 1)
            lefty = np.array(lefty).reshape(-1, 1)
            rightx = np.array(rightx).reshape(-1, 1)
            righty = np.array(righty).reshape(-1, 1)

            # print(len(leftx))
            # iris_length.append(len(leftx))
            eye_landmarks = np.stack((leftx, lefty, rightx, righty), axis=1)
            eye_landmarks = eye_landmarks.reshape(-1, 4)
            # print(eye_landmarks)

            # 写入csv
            landmarks = pd.DataFrame(columns=name, data=eye_landmarks)
            # print(landmarks)
            # print(face_x)
            landmarks.to_csv(detail_path + '/iris_location.csv', encoding='gbk')

            # 情况landmarks列表
            eye_landmarks = []
            leftx = []
            lefty = []
            rightx = []
            righty = []
    # img_path = './Dataset/healthy/00002-0101/2021-05-17-15-18-26.png'

    print(max(iris_length))

