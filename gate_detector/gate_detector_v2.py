#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
案B: YOLO検出 + HSV色分類によるゲート検出
- YOLOでゲート形状を検出し、bbox内のHSV多数決で赤/青/黄を分類
- ストリーム画面で色ボタンを選び、映像内のゲートをクリック → 色分類用の閾値を生成
- 閾値は thresholds.json に保存され、次回起動時に自動読込
- 各ゲートの cx(bbox中心x), area(bbox面積) を映像に重畳表示

実行時は source ~/yolo-env/bin/activate で仮想環境を有効化してから実行
"""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import cv2
import json
import numpy as np
import threading
import time
from flask import Flask, Response, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

CAM_W, CAM_H = 640, 480
ROI_HALF = 15          # サンプリング窓の半径(px)
H_DELTA = 10           # Hの許容幅（狭く）
S_MIN_MARGIN = 60      # サンプル中央値からのS下限マージン（広く）
V_MIN_MARGIN = 60      # 同V下限
CONF_TH = 0.5          # ゲート検出精度の閾値
MIN_COLOR_RATIO = 0.05 # bbox面積に対する色ピクセルの下限比(下回れば色不明)
TARGET_COLOR = None    # 制御時の目標色。Noneなら絞り込み表示なし(テスト用)
THRESH_FILE = "thresholds.json"
model = YOLO("/home/pi/gate-yolo/best_ncnn_model", task="detect")

# 推論のウォームアップ
dummy = np.zeros((480, 640, 3), dtype=np.uint8)
model(dummy, imgsz=320, verbose=False)

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

def detect(frame):
    # フレームからゲートを検出
    results = model(frame, imgsz=320, conf=CONF_TH, verbose=False)
    out = []    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(CAM_W, x2); y2 = min(CAM_H, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        color, color_px = classify_color(frame, x1, y1, x2, y2)
        # ROI面積の5%に満たない色は「たまたま最多だっただけ」と見なし色不明にする。
        # color_pxは0にせず実測値を残す（min_pxと並べて表示し閾値調整の材料にするため）
        min_px = int((x2 - x1) * (y2 - y1) * MIN_COLOR_RATIO)
        if color_px < min_px:
            color = None
        conf = float(box.conf[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        area = (x2 - x1) * (y2 - y1)
        out.append({"color": color, "cx": cx, "cy": cy, "area": area,
                    "bbox": (x1, y1, x2, y2), "conf": conf,
                    "color_px": color_px, "min_px": min_px})
    return out


def pick_target(detections, target_color):
    """制御用の絞り込み: 目標色のうち最大areaのもの(=最も近いゲート)を1つ返す。
       複数ゲートが視野に入っても、重複検出が出ても、ここで1つに潰れる。"""
    cands = [d for d in detections if d["color"] == target_color]
    if not cands:
        return None
    return max(cands, key=lambda d: d["area"])

def classify_color(frame, x1, y1, x2, y2):
    with lock:
        th = dict(thresholds)
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    best_color, best_count = None, 0
    for color, ranges in th.items():
        mask = None
        for lo, up in ranges:
            m = cv2.inRange(hsv, lo, up)
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        count = cv2.countNonZero(mask)
        if count > best_count:
            best_color, best_count = color, count
    return best_color, best_count

camera = cv2.VideoCapture("/dev/video1", cv2.CAP_V4L2)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

latest_frame = None  # キャリブレーション用に最新フレームを保持
frame_lock = threading.Lock()

def generate_frames():
    global latest_frame
    prev_t = time.time()
    fps = 0.0
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        now = time.time()
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # 検出は「描画前の生フレーム」に対して行う。
        # frameへ描いた線や文字がbbox内に入るとHSV色分類に混入するため、
        # 検出用(raw)と表示用(frame)を分離しておく。
        raw = frame.copy()
        with frame_lock:
            latest_frame = raw               # 描画前を保持（教師データ汚染防止）

        cv2.line(frame, (CAM_W // 2, 0), (CAM_W // 2, CAM_H), (128, 128, 128), 1)

        try:
            detections = detect(raw)
        except Exception as e:
            detections = []
            cv2.putText(frame, f"DETECT ERROR: {type(e).__name__}", (8, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        target = pick_target(detections, TARGET_COLOR) if TARGET_COLOR else None

        for r in detections:
            x1, y1, x2, y2 = r["bbox"]
            c = DRAW_COLOR.get(r["color"], (200, 200, 200))
            is_target = target is not None and r is target
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 4 if is_target else 2)
            cv2.circle(frame, (r["cx"], r["cy"]), 5, c, -1)
            err = r["cx"] - CAM_W // 2
            ty = y1 - 8 if y1 > 30 else y2 + 16
            head = "TARGET " if is_target else ""
            cv2.putText(frame, f"{head}{r['color']} {r['conf']:.2f} err={err}",
                        (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
            cv2.putText(frame, f"area={r['area']} px={r['color_px']}/{r['min_px']}",
                        (x1, ty + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

        cv2.putText(frame, f"{fps:.1f} FPS", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 未校正の色があると全て色不明(灰)になるため、その旨を明示する
        with lock:
            missing = [c for c in DRAW_COLOR if c not in thresholds]
        if missing:
            cv2.putText(frame, f"NOT CALIBRATED: {','.join(missing)}", (8, CAM_H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

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
