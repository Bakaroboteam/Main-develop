#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
案A: 色抽出ベースのゲート検出（3色同時・クリックキャリブレーション付き）
- ストリーム画面で色ボタンを選び、映像内のゲートをクリック → その点周辺をサンプリングして閾値生成
- 閾値は thresholds.json に保存され、次回起動時に自動読込
- 各色の cx(重心x), area(面積) を映像に重畳表示
"""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import cv2
import json
import numpy as np
import threading
from flask import Flask, Response, request, jsonify

app = Flask(__name__)

CAM_W, CAM_H = 640, 480
ROI_HALF = 15          # サンプリング窓の半径(px)
H_DELTA = 10           # Hの許容幅（狭く）
S_MIN_MARGIN = 60      # サンプル中央値からのS下限マージン（広く）
V_MIN_MARGIN = 60      # 同V下限
MIN_AREA = 500         # ノイズ除去: これ未満の連結成分は無視
DILATE_KERNEL = np.ones((25, 25), np.uint8)  # 継手の分断を埋める膨張カーネル
THRESH_FILE = "thresholds.json"

# 描画用BGR
DRAW_COLOR = {"red": (0, 0, 255), "blue": (255, 0, 0), "yellow": (0, 255, 255)}

# 閾値の読み書き排他
lock = threading.Lock()

# 閾値: color -> list of (lower, upper)  ※赤はH円環またぎで2範囲になり得る
thresholds = {}

def load_thresholds():
    global thresholds
    if os.path.exists(THRESH_FILE):
        with open(THRESH_FILE) as f:
            raw = json.load(f)
        thresholds = {c: [(np.array(lo), np.array(up)) for lo, up in ranges]
                      for c, ranges in raw.items()}

def save_thresholds():
    raw = {c: [[lo.tolist(), up.tolist()] for lo, up in ranges]
           for c, ranges in thresholds.items()}
    with open(THRESH_FILE, "w") as f:
        json.dump(raw, f)

def build_threshold(h_med, s_med, v_med):
    """中央値から閾値範囲を生成。Hは狭く、S/Vは下限のみ広めに取る。
       赤のH円環またぎ(0/180境界)は2範囲に分割して返す。"""
    s_min = max(int(s_med) - S_MIN_MARGIN, 30)
    v_min = max(int(v_med) - V_MIN_MARGIN, 30)
    h_lo, h_hi = int(h_med) - H_DELTA, int(h_med) + H_DELTA
    ranges = []
    if h_lo < 0:      # 0側にはみ出す → 高H側にも範囲を作る
        ranges.append((np.array([0, s_min, v_min]), np.array([h_hi, 255, 255])))
        ranges.append((np.array([180 + h_lo, s_min, v_min]), np.array([180, 255, 255])))
    elif h_hi > 180:  # 180側にはみ出す
        ranges.append((np.array([h_lo, s_min, v_min]), np.array([180, 255, 255])))
        ranges.append((np.array([0, s_min, v_min]), np.array([h_hi - 180, 255, 255])))
    else:
        ranges.append((np.array([h_lo, s_min, v_min]), np.array([h_hi, 255, 255])))
    return ranges

def detect(frame, ranges):
    """1色分の検出: mask→膨張→最大成分のbboxと重心・面積(膨張前mask基準)を返す"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = None
    for lo, up in ranges:
        m = cv2.inRange(hsv, lo, up)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    dilated = cv2.dilate(mask, DILATE_KERNEL)   # 継手で分断された3本を連結
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    target = max(contours, key=cv2.contourArea)
    if cv2.contourArea(target) < MIN_AREA:
        return None
    x, y, w, h = cv2.boundingRect(target)
    # 面積と重心は膨張前maskのbbox内画素で算出（膨張の水増しを排除）
    sub = mask[y:y+h, x:x+w]
    area = int(cv2.countNonZero(sub))
    if area < MIN_AREA:
        return None
    mm = cv2.moments(sub, binaryImage=True)
    cx = x + int(mm["m10"] / mm["m00"])
    cy = y + int(mm["m01"] / mm["m00"])
    return {"cx": cx, "cy": cy, "area": area, "bbox": (x, y, w, h)}

camera = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

latest_frame = None  # キャリブレーション用に最新フレームを保持
frame_lock = threading.Lock()

def generate_frames():
    global latest_frame
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        with frame_lock:
            latest_frame = frame.copy()
        with lock:
            th = dict(thresholds)
        for color, ranges in th.items():
            r = detect(frame, ranges)
            if r is None:
                continue
            x, y, w, h = r["bbox"]
            c = DRAW_COLOR[color]
            cv2.rectangle(frame, (x, y), (x + w, y + h), c, 2)
            cv2.circle(frame, (r["cx"], r["cy"]), 5, c, -1)
            cv2.line(frame, (CAM_W // 2, 0), (CAM_W // 2, CAM_H), (128, 128, 128), 1)
            err = r["cx"] - CAM_W // 2
            cv2.putText(frame, f"{color} cx={r['cx']} err={err} area={r['area']}",
                        (x, max(y - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buffer.tobytes() + b"\r\n")

@app.route("/")
def index():
    return """<html><body style="font-family:sans-serif">
<div>
  <button onclick="sel('red')">赤を校正</button>
  <button onclick="sel('blue')">青を校正</button>
  <button onclick="sel('yellow')">黄を校正</button>
  <span id="st">色ボタンを押してから映像内のゲートをクリック</span>
</div>
<img id="v" src="/video" style="cursor:crosshair">
<script>
let color = null;
function sel(c){ color = c; document.getElementById('st').textContent = c + ' を校正中: ゲートをクリック'; }
document.getElementById('v').onclick = async (e) => {
  if(!color){ return; }
  const img = e.target, rect = img.getBoundingClientRect();
  const x = Math.round((e.clientX - rect.left) * img.naturalWidth / rect.width);
  const y = Math.round((e.clientY - rect.top) * img.naturalHeight / rect.height);
  const res = await fetch(`/calibrate/${color}?x=${x}&y=${y}`, {method:'POST'});
  const j = await res.json();
  document.getElementById('st').textContent = j.msg;
  color = null;
};
</script></body></html>"""

@app.route("/calibrate/<color>", methods=["POST"])
def calibrate(color):
    if color not in DRAW_COLOR:
        return jsonify(msg="不明な色"), 400
    x, y = int(request.args["x"]), int(request.args["y"])
    with frame_lock:
        frame = None if latest_frame is None else latest_frame.copy()
    if frame is None:
        return jsonify(msg="フレーム未取得"), 500
    x0, x1 = max(x - ROI_HALF, 0), min(x + ROI_HALF, CAM_W)
    y0, y1 = max(y - ROI_HALF, 0), min(y + ROI_HALF, CAM_H)
    roi = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    h_med = int(np.median(roi[:, :, 0]))
    s_med = int(np.median(roi[:, :, 1]))
    v_med = int(np.median(roi[:, :, 2]))
    with lock:
        thresholds[color] = build_threshold(h_med, s_med, v_med)
        save_thresholds()
    return jsonify(msg=f"{color} 校正完了 H={h_med} S={s_med} V={v_med}")

@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    load_thresholds()
    app.run(host="0.0.0.0", port=8000, threaded=True)
