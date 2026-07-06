#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import cv2
from flask import Flask, Response

app = Flask(__name__)

# トラックバーのためのウィンドウを作成
cv2.namedWindow("OpenCV Window")
def nothing(x):
    pass

# hsv閾値
cv2.createTrackbar("H_min", "OpenCV Window", 0, 179, nothing)   # 色相最小閾値(0~179)
cv2.createTrackbar("H_max", "OpenCV Window", 128, 179, nothing) # 色相最大閾値(128~179)
cv2.createTrackbar("S_min", "OpenCV Window", 128, 255, nothing) # 彩度最小閾値(128~255)
cv2.createTrackbar("S_max", "OpenCV Window", 255, 255, nothing) # 彩度最大閾値(S_min~255)
cv2.createTrackbar("V_min", "OpenCV Window", 128, 255, nothing) # 明度最小閾値(128~255)
cv2.createTrackbar("V_max", "OpenCV Window", 255, 255, nothing) # 明度最大閾値(V_min~255)

camera = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) 

def generate_frames():
    while True:

        # フレームを取得
        ret, frame = camera.read()
        if not ret:
            break
        
        # 画像処理
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # トラックバーの値を取る
        h_min = cv2.getTrackbarPos("H_min", "OpenCV Window")
        h_max = cv2.getTrackbarPos("H_max", "OpenCV Window")
        s_min = cv2.getTrackbarPos("S_min", "OpenCV Window")
        s_max = cv2.getTrackbarPos("S_max", "OpenCV Window")
        v_min = cv2.getTrackbarPos("V_min", "OpenCV Window")
        v_max = cv2.getTrackbarPos("V_max", "OpenCV Window")

        # inRange関数で範囲指定２値化 -> マスク画像用
        mask_image = cv2.inRange(hsv_image, (h_min, s_min, v_min), (h_max, s_max, v_max))
        
        # bitwise_andで元画像にマスクをかける -> マスクされた部分の色だけ残る
        result_image = cv2.bitwise_and(hsv_image, hsv_image, mask=mask_image)

        ok, buffer = cv2.imencode('.jpg', result_image) # フレームをjpegにエンコード
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