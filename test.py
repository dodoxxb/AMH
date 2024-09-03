import os
import mediapipe as mp
import cv2 as cv
import pandas as pd
import random
import numpy as np


list12 = [1,2,3,4,5]
list2 = [9,8,7,6]
random_num = random.sample(range(0,321), 95)
print(random_num)
labels = []
labels.append(np.zeros(323).tolist())
labels.append(np.ones(95).tolist())
print(labels)
list_copy = list(list12[0:3])
print(list_copy)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

img_path = './Dataset/healthy/00002-0101/2021-05-17-15-18-19.png'
with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
    image = cv.imread(img_path)
    results = face_mesh.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        print("No landmarks")

    # output face with 468 landmarks
    annotated_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:
        # print('face_landmarks:', face_landmarks)
        # mp_drawing.draw_landmarks(
        #     image=annotated_image,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_TESSELATION,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles
        #         .get_default_face_mesh_tesselation_style())
        # mp_drawing.draw_landmarks(
        #     image=annotated_image,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_CONTOURS,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles
        #         .get_default_face_mesh_contours_style())
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