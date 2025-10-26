from flask import render_template, Blueprint, Response
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import threading
from flask_socketio import SocketIO, emit

from collections import deque
from app.utils import mediapipeDetection, drawStyledLandmarks, extractLandmarks

# Camera
# Instructions/Manual, Mpst of the load is here, Translation and history(collapsable)

bp = Blueprint("pageNames", __name__)
mpHolistic = mp.solutions.holistic     #Holistic Model
# theres an instantiation for socketio here

@bp.route('/')
def home():
    return render_template("pageNames/home.html")

@bp.route('/about')
def about():
    return render_template("pageNames/about.html")

def getStream():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success,frame = cap.read()
        if not success:
            print("Could not read camera")
            break
        
        frame = cv2.resize(frame, (640, 480))  # Reduce frame size for efficiency

        _,buffer = cv2.imencode('.jpg',frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        time.sleep(1/30) # frame rate control
    # there are 2 problems here:
    #   if exiting: there is no way to cap.release(), cv2.destroyAllWindows() which could cause ram leaks
    #   socketing having an image feed conflicts with trying to return something from here
    #   remember to add a cdn url for socketio: <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    #   a more secure cdn url: <script src="https://cdn.socket.io/4.5.4/socket.io.min.js" integrity="sha384-/KNQL8Nu5gCHLqwqfQjA689Hhoqgi2S84SNUxC3roTe4EhJ9AfLkp8QiQcU8AMzI" crossorigin="anonymous"></script>


def getStreamData(socketio):
    pass

@bp.route("/runCamera")
def runCamera():
    return Response(getStream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@bp.route('/camera')
def camera():
    return render_template("pageNames/camera.html")