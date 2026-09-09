#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NCNN モデルの出力が壊れているかを、1枚の画像で確かめる。

conf がちょうど 1.0 で 100個超の枠が出るのは正常な YOLO の挙動ではない。
同じ画像を NCNN と元の .pt に通して比べれば、エクスポートが原因か切り分けられる。

    python3 model_probe.py                       # カメラから1枚取って判定
    python3 model_probe.py --image shot.jpg      # 画像ファイルで判定
    python3 model_probe.py --pt ~/runs/detect/train6/weights/best.pt
"""

import argparse
import os

import cv2
import numpy as np
from ultralytics import YOLO

NCNN_PATH = "/home/pi/gate-yolo/best_ncnn_model"
DEVICE = "/dev/video1"


def grab():
    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(10):
        ok, f = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("カメラから取得できない")
    cv2.imwrite("probe_frame.jpg", f)
    print("フレームを probe_frame.jpg に保存した")
    return f


def run(label, model, frame, imgsz, conf):
    r = model.predict(frame, imgsz=imgsz, conf=conf, max_det=300, verbose=False)[0]
    b = r.boxes
    n = 0 if b is None else len(b)
    print("  %-38s 検出 %4d 個" % (label, n), end="")
    if n:
        cf = b.conf.cpu().numpy()
        xy = b.xyxy.cpu().numpy()
        print("  conf 最小 %.3f / 最大 %.3f / ちょうど1.0 が %d 個"
              % (cf.min(), cf.max(), int((cf >= 0.9999).sum())))
        for i in range(min(n, 3)):
            x1, y1, x2, y2 = xy[i]
            print("      #%d conf=%.3f bbox=[%d,%d,%d,%d]"
                  % (i, cf[i], x1, y1, x2, y2))
    else:
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--pt", default="", help="比較する .pt のパス")
    ap.add_argument("--conf", type=float, default=0.60)
    a = ap.parse_args()

    frame = cv2.imread(a.image) if a.image else grab()
    if frame is None:
        raise SystemExit("画像を読めない: %s" % a.image)
    print("入力: %dx%d  conf=%.2f" % (frame.shape[1], frame.shape[0], a.conf))
    print("=" * 78)

    print("NCNN (%s)" % NCNN_PATH)
    ncnn = YOLO(NCNN_PATH, task="detect")
    run("imgsz=[256,320]（metadata と一致）", ncnn, frame, [256, 320], a.conf)
    run("imgsz=320（不一致）",                ncnn, frame, 320, a.conf)
    run("imgsz=[256,320] conf=0.01",          ncnn, frame, [256, 320], 0.01)

    if a.pt and os.path.exists(a.pt):
        print("-" * 78)
        print("PyTorch (%s)" % a.pt)
        pt = YOLO(a.pt)
        run("imgsz=[256,320]", pt, frame, [256, 320], a.conf)
        run("imgsz=320",       pt, frame, 320, a.conf)
    elif a.pt:
        print("  .pt が見つからない: %s" % a.pt)
    print("=" * 78)
    print("判定の目安:")
    print("  NCNN が 100個超・conf ちょうど 1.0 → エクスポートが壊れている。再エクスポート")
    print("  .pt が正常で NCNN だけ壊れる      → 同上")
    print("  両方とも同じ結果                  → モデルではなく入力側の問題")


if __name__ == "__main__":
    main()
