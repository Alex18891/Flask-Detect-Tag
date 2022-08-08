import pyglet
import numpy as np
from flask import Flask,render_template,Response
import cv2
import keyboard
import yolotrainsimbolos
from yolov5 import YOLOv5
from flaskwebgui import FlaskUI


app = Flask(__name__)
model_path = "C:/Users/Alexandre/yolov5/runs/train/exp2/weights/best.pt"
device = "cuda:0" # or "cpu
yolov5 = YOLOv5(model_path, device)
player = pyglet.media.Player()
src = pyglet.media.load('C:/Users/Alexandre/PycharmProjects/pythonProject/pythonProject/audio/Inicio.mp3')
player.queue(src)
app.secret_key = "secret key"

@app.route("/process",methods=['POST','GET'])
def key():
    a = 1
    while True:
        a = a + 1
        print(a)
        if keyboard.is_pressed('enter'):
            Response(readout())
            return render_template('index.html')


@app.route("/")
def page1():
    return render_template('page1.html')

@app.route("/ask",methods=['POST','GET'])
def ask():
    player.play()
    while True:
        print("Tempo",player.time)
        if (player.time > 37) or keyboard.is_pressed('enter'):
            player.pause()
            return render_template('index.html')

@app.route("/page1")
def index():
    player.pause()
    return render_template('index.html')

@app.route("/result",methods=['POST','GET'])
def result():
    Response(readout())
    return render_template('index.html')

def frames():
    img = cv2.VideoCapture(0)
    while True:
        ret, frame = img.read()
        if not ret:
            break
        else:
            results = yolov5.predict(frame)
            frame = np.squeeze(results.render())
            frame = cv2.copyMakeBorder(frame, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[186, 82, 15])
            ret,buffer = cv2.imencode('.JPEG', frame)
            frame = buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n'+frame + b'\r\n')
    img.release()

@app.route('/video_feed')
def video_feed():
    return Response(frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def readout():
    img = cv2.VideoCapture(0)
    ret, frame = img.read()
    if not ret:
        return None
    yolotrainsimbolos.test_img(frame)
    img.release()

if __name__ == "__main__":
    FlaskUI(app, width=1920, height=1080).run()

