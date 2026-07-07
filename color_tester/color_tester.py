#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
from flask import Flask, Response, request, jsonify

app = Flask(__name__)

# HSV閾値をグローバルに保持（トラックバーの代わり）
hsv = {"h_min": 0, "h_max": 179, "s_min": 128, "s_max": 255,
       "v_min": 128, "v_max": 255}

camera = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

def generate_frames():
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        # RGB2BGR変換は削除。frameはすでにBGR
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = (hsv["h_min"], hsv["s_min"], hsv["v_min"])
        upper = (hsv["h_max"], hsv["s_max"], hsv["v_max"])
        mask = cv2.inRange(hsv_image, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)  # BGRのまま表示用に

        ok, buffer = cv2.imencode('.jpg', result)
        if not ok:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/set', methods=['POST'])
def set_hsv():
    # JSから送られた値でグローバル変数を更新
    for k, v in request.get_json().items():
        if k in hsv:
            hsv[k] = int(v)
    return jsonify(hsv)

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
<html><body>
  <img src="/video"><br>
  <div id="sliders"></div>
  <script>
    const params = [
      ["h_min",179,0],["h_max",179,179],
      ["s_min",255,128],["s_max",255,255],
      ["v_min",255,128],["v_max",255,255]
    ];
    const box = document.getElementById("sliders");
    params.forEach(([name, max, val]) => {
      const label = document.createElement("label");
      label.textContent = name + ": ";
      const span = document.createElement("span");
      span.textContent = val;
      const s = document.createElement("input");
      s.type = "range"; s.min = 0; s.max = max; s.value = val;
      s.oninput = () => {
        span.textContent = s.value;
        fetch("/set", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({[name]: s.value})
        });
      };
      box.append(label, s, span, document.createElement("br"));
    });
  </script>
</body></html>'''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True)