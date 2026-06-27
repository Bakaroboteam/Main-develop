#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import cv2
import signal
from pyzbar import pyzbar

running = True
last_qr_data = None

def signal_handler(sig, frame):
    """
    ctrl+cを受け取ったら呼び出される
    running フラグを False に設定し、メインループを終了する
    """
    global running
    print("\nSignal received, shutting down...")
    running = False

def main():
    """
    カメラから映像を取得し、ウィンドウに表示するメイン関数
    """

    global running, last_qr_data

    print("camera monitor starting...\n")

    #１．シグナルハンドラの設定
    #ctrl+cを受け取ったら signal_handler() が呼ばれる
    signal.signal(signal.SIGINT, signal_handler)

    #２．メインループ
    #カメラからフレームをキャプチャし、ウィンドウに表示
    while running:
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            print("Error: Could not open camera")
            break

        ret, frame = camera.read()

        if ret == True:
            cv2.imshow("camera monitor", frame)

        else:
            print("Error: Could not read frame from camera")
            break

        time.sleep(0.1)

    print("\nmonitor shutting down...")
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sys.exit(main())