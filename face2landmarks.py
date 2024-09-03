import os
import mediapipe as mp
import cv2 as cv
import pandas as pd
import numpy as np

def get_landmarks(img_path):
    x = []
    y = []
    z = []

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:

        image = cv.imread(img_path)
        results = face_mesh.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            print("No landmarks")
            return 0, 0, 0

        for i in range(468):
            x.append(results.multi_face_landmarks[0].landmark[i].x)
            y.append(results.multi_face_landmarks[0].landmark[i].y)
            z.append(results.multi_face_landmarks[0].landmark[i].z)
        # print(x, y, z)

        # output face with 468 landmarks
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            # print('face_landmarks:', face_landmarks)
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
        # cv.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
        # cv.imshow("Image", annotated_image)
        # cv.waitKey(0)
        # cv.destroyWindow()

        # x = np.array(x).reshape(468, 1)
        # y = np.array(y).reshape(468, 1)
        # z = np.array(z).reshape(468, 1)

        return x,y,z

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
    name = ['x', 'y', 'z']
    face_landmarks = []
    face_x = []
    face_y = []
    face_z = []

    # 遍历dataset文件夹
    dataset_class = os.listdir('./Dataset')
    print(dataset_class)

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
                    # print(img)
                    # 判断是否有人脸
                    judge = judge_face(detail_path + '/' + img)
                    # 有人脸则获取人脸区域的468个标记点
                    if judge == 1:
                        x, y, z = get_landmarks(detail_path + '/' + img)
                        # 如果没有返回468个标记点
                        if x == 0:
                            continue
                        face_x.append(x)
                        face_y.append(y)
                        face_z.append(z)
                        # print(np.array(face_landmarks).shape)
                        # print(face_landmarks.shape)
                    else:
                        continue
                else:
                    continue
            face_x = np.array(face_x).reshape(-1, 1)
            face_y = np.array(face_y).reshape(-1, 1)
            face_z = np.array(face_z).reshape(-1, 1)
            # print(np.array(face_x).shape)
            # face_landmarks = np.array(face_landmarks).reshape(-1,3)
            face_landmarks = np.stack((face_x, face_y, face_z), axis=1)
            face_landmarks = face_landmarks.reshape(-1, 3)
            # print(np.array(face_landmarks).shape)

            # 写入csv
            landmarks = pd.DataFrame(columns=name, data=face_landmarks)
            print(landmarks)
            # print(face_x)
            landmarks.to_csv(detail_path + '/face_landmarks.csv', encoding='gbk')

            #情况landmarks列表
            face_landmarks = []
            face_x = []
            face_y = []
            face_z = []


