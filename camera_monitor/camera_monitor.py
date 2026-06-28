#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import cv2
from flask import Flask, Response

app = Flask(__name__)
running = True

camera = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 

def generate_frames():
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        # フレームをJPEGにエンコード
        ok, buffer = cv2.imencode('.jpg', frame)
        if not ok:
            continue
        jpg = buffer.tobytes()
        # 境界線 + ヘッダ + JPEG本体 を yield で送り続ける
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')

@app.route('/')
def index():
    # img タグの src にストリームを指定するだけで表示される
    return '<html><body><img src="/video"></body></html>'

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 0.0.0.0 で全インターフェース待受（VNC内ブラウザから見るなら localhost でも可）
    app.run(host='0.0.0.0', port=8000, threaded=True)