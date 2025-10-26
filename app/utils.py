import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

mpHolistic = mp.solutions.holistic     #Holistic Model
mpDrawing = mp.solutions.drawing_utils #Drawing Utilities


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# if i get more errors or somethings feel off
# try to rollback to mediapipe 10.9 if this gets bad
# Feedback manager requires a model with a single signature inference

#cap.release and destroywindow is not properly implemented

#projection is not working (styledlandmarks)

#abrupt exit may cause problems


def mediapipeDetection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, results

def drawStyledLandmarks(image, results):
    mpDrawing.draw_landmarks(
        image, results.face_landmarks, mpHolistic.FACEMESH_TESSELATION,
        mpDrawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        mpDrawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    )
    mpDrawing.draw_landmarks(
        image, results.pose_landmarks, mpHolistic.POSE_CONNECTIONS,
        mpDrawing.DrawingSpec(color=(210,166,205), thickness=1, circle_radius=1),
        mpDrawing.DrawingSpec(color=(150,156,170), thickness=1, circle_radius=1)
    )
    mpDrawing.draw_landmarks(
        image, results.left_hand_landmarks, mpHolistic.HAND_CONNECTIONS,
        mpDrawing.DrawingSpec(color=(142,202,230), thickness=1, circle_radius=1),
        mpDrawing.DrawingSpec(color=(33,158,188), thickness=1, circle_radius=1)
    )
    mpDrawing.draw_landmarks(
        image, results.right_hand_landmarks, mpHolistic.HAND_CONNECTIONS,
        mpDrawing.DrawingSpec(color=(148,190,71), thickness=1, circle_radius=1),
        mpDrawing.DrawingSpec(color=(91,168,160), thickness=1, circle_radius=1)
    )

def extractLandmarks(results):
    faceLmk = np.zeros((468,3), dtype=np.float32)
    if results.face_landmarks:
        for i, landmark in enumerate(results.face_landmarks.landmark):
            faceLmk[i] = np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)

    poseLmk = np.zeros((33,4), dtype=np.float32)
    if results.pose_landmarks:
        offset = 0
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            poseLmk[i] = np.array([landmark.x, landmark.y, landmark.z, landmark.visibility], dtype=np.float32)
    
    leftHLmk = np.zeros((21,3), dtype=np.float32)
    if results.left_hand_landmarks:
        for i, landmark in enumerate(results.left_hand_landmarks.landmark):
            leftHLmk[i] = np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)

    rightHLmk = np.zeros((21,3), dtype=np.float32)
    if results.right_hand_landmarks:
        for i, landmark in enumerate(results.right_hand_landmarks.landmark):
            rightHLmk[i] = np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)

    return np.concatenate((faceLmk.flatten(), poseLmk.flatten(), leftHLmk.flatten(), rightHLmk.flatten()))
