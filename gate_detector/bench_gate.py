#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pi4 上でゲート検出の所要時間を段階別に計測する。

gate_detector_v4.py には一切触らない独立スクリプト。Flask を起動しないため、
これ自体が「配信 OFF」の条件になる。v4 の実測 fps との差が配信の負荷。

使い方（Pi4 上、venv を有効にしてから）:
    python3 bench_gate.py                    # 既定 200 フレーム、imgsz=320
    python3 bench_gate.py --imgsz 256 320    # 矩形入力で比較
    python3 bench_gate.py --no-color         # 色分類を外し YOLO 単体を見る
    python3 bench_gate.py -n 400
"""

import argparse
import json
import os
import subprocess
import time

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_DIR = "/home/pi/gate-yolo/best_ncnn_model"
THRESH_FILE = "thresholds.json"
CAM_W, CAM_H = 640, 480
CONF_TH = 0.60          # train-6 の F1 ピーク
MIN_COLOR_RATIO = 0.05

# vcgencmd get_throttled のビット定義
THROTTLE_BITS = [
    (0,  "低電圧を検出（現在）"),
    (1,  "ARM 周波数を制限（現在）"),
    (2,  "スロットリング中（現在）"),
    (3,  "ソフト温度制限が作動中（現在）"),
    (16, "低電圧が発生した（過去）"),
    (17, "ARM 周波数の制限が発生した（過去）"),
    (18, "スロットリングが発生した（過去）"),
    (19, "ソフト温度制限が作動した（過去）"),
]


def vcgencmd(*args):
    try:
        return subprocess.check_output(["vcgencmd", *args], text=True).strip()
    except Exception as e:
        return "取得失敗 (%s)" % type(e).__name__


def report_power(label):
    raw = vcgencmd("get_throttled")
    print("[%s] %s / %s / arm=%s" % (label, raw, vcgencmd("measure_temp"),
                                     vcgencmd("measure_clock", "arm")))
    if "=" in raw:
        try:
            v = int(raw.split("=")[1], 16)
        except ValueError:
            return
        hits = [name for bit, name in THROTTLE_BITS if v & (1 << bit)]
        if hits:
            for h in hits:
                print("        - " + h)
        else:
            print("        - 給電・温度ともに問題なし")


def load_thresholds():
    if not os.path.exists(THRESH_FILE):
        print("警告: %s が無いため色分類は空振りする" % THRESH_FILE)
        return {}
    with open(THRESH_FILE) as f:
        raw = json.load(f)
    return {c: [(np.array(lo), np.array(up)) for lo, up in r] for c, r in raw.items()}


def classify_color(frame, th, x1, y1, x2, y2):
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    best, best_n = None, 0
    for color, ranges in th.items():
        mask = None
        for lo, up in ranges:
            m = cv2.inRange(hsv, lo, up)
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        n = cv2.countNonZero(mask)
        if n > best_n:
            best, best_n = color, n
    return best, best_n


def stats(name, xs):
    if not xs:
        return
    s = sorted(xs)
    print("  %-14s 平均 %6.1f  中央 %6.1f  p90 %6.1f  最大 %6.1f ms"
          % (name, sum(s) / len(s), s[len(s) // 2], s[int(len(s) * 0.9)], s[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=200, help="計測フレーム数")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[320])
    ap.add_argument("--no-color", action="store_true", help="色分類を行わない")
    ap.add_argument("--warmup", type=int, default=10)
    a = ap.parse_args()
    imgsz = a.imgsz[0] if len(a.imgsz) == 1 else a.imgsz

    print("=" * 62)
    report_power("開始時")
    print("  OpenCV スレッド数: %d" % cv2.getNumThreads())

    meta = os.path.join(MODEL_DIR, "metadata.yaml")
    if os.path.exists(meta):
        with open(meta) as f:
            for line in f:
                if line.startswith(("imgsz", "task", "batch", "stride")):
                    print("  metadata: " + line.rstrip())
    else:
        print("  metadata.yaml が見つからない: " + meta)
    print("  推論時の指定 imgsz: %s" % (imgsz,))
    print("=" * 62)

    model = YOLO(MODEL_DIR, task="detect")
    th = {} if a.no_color else load_thresholds()

    cam = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not cam.isOpened():
        raise SystemExit("カメラを開けない")

    model(np.zeros((CAM_H, CAM_W, 3), np.uint8), imgsz=imgsz, verbose=False)
    for _ in range(a.warmup):
        cam.read()

    t_read, t_infer, t_color, t_loop = [], [], [], []
    sp_pre, sp_inf, sp_post = [], [], []
    n_det = []
    t_start = time.time()

    for i in range(a.n):
        l0 = time.perf_counter()

        r0 = time.perf_counter()
        ok, frame = cam.read()
        r1 = time.perf_counter()
        if not ok:
            continue

        res = model(frame, imgsz=imgsz, conf=CONF_TH, verbose=False)
        i1 = time.perf_counter()

        boxes = res[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(CAM_W, x2), min(CAM_H, y2)
            if x2 > x1 and y2 > y1 and th:
                classify_color(frame, th, x1, y1, x2, y2)
        c1 = time.perf_counter()

        t_read.append((r1 - r0) * 1e3)
        t_infer.append((i1 - r1) * 1e3)
        t_color.append((c1 - i1) * 1e3)
        t_loop.append((c1 - l0) * 1e3)
        n_det.append(len(boxes))

        sp = res[0].speed          # ultralytics 内部の内訳(ms)
        sp_pre.append(sp.get("preprocess", 0.0))
        sp_inf.append(sp.get("inference", 0.0))
        sp_post.append(sp.get("postprocess", 0.0))

        if (i + 1) % 50 == 0:
            print("  ... %d/%d" % (i + 1, a.n))

    elapsed = time.time() - t_start
    cam.release()

    print("=" * 62)
    print("計測 %d フレーム / %.1f 秒 → 実効 %.2f fps" % (len(t_loop), elapsed,
                                                    len(t_loop) / elapsed))
    print("  1フレームあたりの内訳")
    stats("カメラ読取", t_read)
    stats("YOLO 推論", t_infer)
    stats("色分類", t_color)
    stats("ループ合計", t_loop)
    print("  ultralytics の内訳（YOLO 推論の中身）")
    stats("前処理", sp_pre)
    stats("推論本体", sp_inf)
    stats("後処理", sp_post)
    print("  検出数 平均 %.2f 個/フレーム" % (sum(n_det) / max(len(n_det), 1)))
    print("=" * 62)
    report_power("終了時")


if __name__ == "__main__":
    main()
