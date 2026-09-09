#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
カメラの配信レートだけを、設定を1つずつ変えて測る。

v4 は 4.5 fps、bench は 8.7 fps 以上出ていた。両者のカメラ設定の差は
  - configure_camera()（v4l2-ctl で露出・WB を固定）
  - CAP_PROP_FPS = 15
  - CAP_PROP_BUFFERSIZE = 1
  - CAP_PROP_AUTO_EXPOSURE / AUTO_WB
の4点。どれが効いているかを潰す。

**v4 を止めてから実行すること**（カメラは排他）。

    python3 cam_probe.py            # 4構成を順に測る
    python3 cam_probe.py --load     # NCNN を別スレッドで回しながら測る（CPU競合の確認）
"""

import argparse
import subprocess
import threading
import time

import cv2
import numpy as np

DEVICE = "/dev/video0"
CAM_W, CAM_H = 640, 480
N = 100

_stop = threading.Event()


def v4l2_fix_exposure():
    for name, val in [("auto_exposure", 1), ("exposure_time_absolute", 200),
                      ("white_balance_automatic", 0), ("white_balance_temperature", 4600)]:
        subprocess.run(["v4l2-ctl", "-d", DEVICE, "-c", "%s=%s" % (name, val)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def v4l2_auto_exposure():
    for name, val in [("auto_exposure", 3), ("white_balance_automatic", 1)]:
        subprocess.run(["v4l2-ctl", "-d", DEVICE, "-c", "%s=%s" % (name, val)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def open_cam(set_fps, set_buffersize, cv2_auto):
    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if set_fps:
        cap.set(cv2.CAP_PROP_FPS, 15)
    if set_buffersize:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if cv2_auto:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    return cap


def measure(label, fix_exp, set_fps, set_buf, cv2_auto):
    if fix_exp:
        v4l2_fix_exposure()
    else:
        v4l2_auto_exposure()
    cap = open_cam(set_fps, set_buf, cv2_auto)
    if not cap.isOpened():
        print("  %-34s カメラを開けない" % label)
        return
    for _ in range(10):
        cap.read()
    gaps = []
    t0 = time.time()
    prev = time.perf_counter()
    ok_n = 0
    for _ in range(N):
        ok, _f = cap.read()
        now = time.perf_counter()
        if ok:
            ok_n += 1
            gaps.append((now - prev) * 1e3)
        prev = now
    total = time.time() - t0
    cap.release()
    g = sorted(gaps)
    fourcc = "-"
    print("  %-34s %5.2f fps  間隔 中央 %6.1f / p90 %6.1f ms  (%d/%d)"
          % (label, ok_n / total, g[len(g) // 2], g[int(len(g) * 0.9)], ok_n, N))


def load_worker():
    """NCNN を回し続けて CPU を埋める。カメラ測定と競合させるため。"""
    from ultralytics import YOLO
    m = YOLO("/home/pi/gate-yolo/best_ncnn_model", task="detect")
    dummy = np.zeros((CAM_H, CAM_W, 3), np.uint8)
    while not _stop.is_set():
        m.predict(dummy, imgsz=[256, 320], verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", action="store_true", help="NCNN を並走させる")
    a = ap.parse_args()

    if a.load:
        print("NCNN を別スレッドで並走させます（CPU競合の確認）")
        threading.Thread(target=load_worker, daemon=True).start()
        time.sleep(5)

    print("=" * 78)
    print("各構成 %d フレーム" % N)
    #        ラベル                              露出固定 FPS  BUF  cv2auto
    measure("A bench 相当（何も設定しない）",      False, False, False, False)
    measure("B v4 相当（全部設定）",              True,  True,  True,  True)
    measure("C v4 から BUFFERSIZE を外す",        True,  True,  False, True)
    measure("D v4 から CAP_PROP_FPS を外す",      True,  False, True,  True)
    measure("E v4 から露出固定だけ外す",          False, True,  True,  True)
    print("=" * 78)
    _stop.set()


if __name__ == "__main__":
    main()
