from flask import render_template, Blueprint, Response
import cv2, time, threading
from app.extensions import socketio  # import the shared socketio instance

import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
from app.utils import mediapipeDetection, drawStyledLandmarks, extractLandmarks

import os


bp = Blueprint("pageNames", __name__)
mpHolistic = mp.solutions.holistic     #Holistic Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

camera_lock = threading.Lock()
latest_frame = None
camera_running = False
camera_thread = None

def camera_loop():
    global latest_frame, camera_running

#-------
    labels = np.array(['label1', 'label2', 'label3'])

    predictionContainer = deque(maxlen=55)    #always keeps the newest 30 frames for predictions; clear afterwards: array.clear()
    wordsPredictedContainer = deque(maxlen=5) #change length for polishing
    prediction = np.zeros(len(labels))
    confidenceLevel = 0.8    #depends on how i want this
    frameLimit = 55

#-------
    MODEL_PATH = os.path.join(BASE_DIR, "fslInit.keras")
    model = tf.keras.models.load_model(MODEL_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not accessible")
        return

    print("[INFO] Camera thread started")
    with mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while camera_running:
            success, frame = cap.read()
            if not success:
                print("Failed to read frame")
                break

            frame = cv2.resize(frame, (640, 480))

#  ---------------------------------------------------------------------------------------------------------

            image, results = mediapipeDetection(frame, holistic)
            drawStyledLandmarks(image, results)
            actionLmk = extractLandmarks(results)  #start from here to export

            predictionContainer.append(actionLmk)   #!!!! Be aware that this may cause an error it and may have to use insert(0,lmk)?
            if len(predictionContainer) == frameLimit:  #slows down probably from constant check
                prediction = model.predict(np.expand_dims(predictionContainer, axis=0))[0]
                predictionContainer.clear()

                # print("prediction here:", prediction)
                # prediction here is expected to be [1. 0. 0.]

            if prediction[np.argmax(prediction)] >= confidenceLevel:
                if not wordsPredictedContainer:    #special case
                    wordsPredictedContainer.append(labels[np.argmax(prediction)])
                else:
                    if wordsPredictedContainer[-1] != labels[np.argmax(prediction)]:
                        wordsPredictedContainer.append(labels[np.argmax(prediction)])

            print(wordsPredictedContainer)

            if len(wordsPredictedContainer) == 5: wordsPredictedContainer.clear()



#-----------------------------------------------------------------------------------------------------------

            _, buffer = cv2.imencode('.jpg', frame)

            with camera_lock:
                latest_frame = buffer.tobytes()

            # Example: send data via SocketIO (optional)
            socketio.emit("frame_data", {'wordPredicted': list(wordsPredictedContainer)})

            time.sleep(1 / 30)  # ~30 FPS

        cap.release()
        print("[INFO] Camera thread stopped")

def start_camera_thread():
    global camera_running
    if not camera_running:
        camera_running = True
        t = threading.Thread(target=camera_loop, daemon=True)
        t.start()

def stop_camera_thread():
    global camera_running
    camera_running = False

def generate_stream():
    while True:
        with camera_lock:
            frame = latest_frame
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(1 / 30)

@bp.route('/')
def home():
    return render_template("pageNames/home.html")

@bp.route('/about')
def about():
    return render_template("pageNames/about.html")

@bp.route('/camera')
def camera():
    return render_template("pageNames/camera2.html")

@bp.route("/runCamera")
def run_camera():
    global camera_running, camera_thread
    if not camera_running:
        camera_running = True
        camera_thread = threading.Thread(target=camera_loop)
        camera_thread.start()
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@bp.route("/releaseCamera", methods=["POST"])
def release_camera():
    global camera_running
    camera_running = False
    return "Camera stopped", 200

