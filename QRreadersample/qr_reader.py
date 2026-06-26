#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QRコード認識サーバー - Pythonバージョン（シングルスレッド）

説明：
  このプログラムはUSBカメラからQRコードを読み込み、
  UNIXドメインソケットを通じてASPプロセスに色情報を送信します。
  
特徴：
  - シンプルなシングルスレッド実装（初心者向け）
  - OpenCV + pyzbarでQRコード認識
  - UNIXドメインソケットでASPプロセスと通信
  - ノンブロッキングIOでカメラ読み込みとクライアント通信を交互実行
  - ハンドシェイク検証で接続確認

使用方法：
  1. 依存ライブラリをインストール：
     pip install -r requirements.txt
     
  2. プログラム起動：
     python3 qr_reader.py
"""

import os
import sys
import socket
import time
import signal
import cv2
from pyzbar import pyzbar

# ==========================================
# グローバル定数
# ==========================================
# UNIXドメインソケットの設定
QR_SOCKET_PATH = "/tmp/qr_socket"

# ハンドシェイク用メッセージ
HANDSHAKE_MSG = "QR_SERVER"         # このサーバーの識別メッセージ
HANDSHAKE_ACK = "ASP_QR_CLIENT"     # クライアント（ASP）からの確認メッセージ

BUFFER_SIZE = 1024

# ==========================================
# グローバル変数
# ==========================================
# プログラム実行フラグ
running = True

# 前回検出したQRコード（連続検出を防ぐ）
last_qr_data = None


# ==========================================
# シグナルハンドラ
# ==========================================
def signal_handler(sig, frame):
    """
    シグナルハンドラ - Ctrl+Cなどで呼ばれる
    
    説明：
      プログラムが SIGINT（Ctrl+C）を受け取った時に呼び出されます。
      running フラグを False に設定してメインループを終了させます。
    """
    global running
    print("\nSignal received, shutting down...")
    running = False


# ==========================================
# ソケットサーバーメイン - シングルスレッド版
# ==========================================
def main():
    """
    QRコード認識サーバーのメイン関数 - シングルスレッド版
    
    説明：
      マルチスレッドを使わず、メインループで以下を順序立てて実行：
      1. カメラからQRコード読み込み
      2. クライアントにQRコード送信
      
    シングルスレッドの利点：
      - コードがシンプルで初心者向け
      - スレッド間通信のロジックが不要
      - デバッグが容易
      
    処理の流れ：
      1. シグナルハンドラ設定
      2. ソケット作成・バインド・リッスン
      3. クライアント接続待機（accept）
      4. ハンドシェイク処理
      5. メインループ：
         - カメラからフレームキャプチャ
         - QRコード認識
         - クライアントに送信
      6. クライアント切断時にループを抜ける
    """
    
    global running, last_qr_data
    
    print("QR Reader Server starting...\n")
    
    # ==========================================
    # ステップ1：シグナルハンドラ設定
    # ==========================================
    # Ctrl+C（SIGINT）を受け取ったら signal_handler() が呼ばれる
    signal.signal(signal.SIGINT, signal_handler)
    
    # ==========================================
    # ステップ2：既存のソケットファイルを削除
    # ==========================================
    # 前回の実行が異常終了した場合、QR_SOCKET_PATH がファイルとして残っている
    # 可能性があります。新規作成前に削除します。
    if os.path.exists(QR_SOCKET_PATH):
        os.remove(QR_SOCKET_PATH)
    
    # ==========================================
    # ステップ3：UNIXドメインソケット作成
    # ==========================================
    # socket.AF_UNIX：UNIXドメインソケット（ファイルベース）
    # socket.SOCK_STREAM：確実な順序でのデータ送受信
    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    
    try:
        # ==========================================
        # ステップ4：ソケットをバインド（名前付け）
        # ==========================================
        # bind() - ソケットに名前（ファイルパス）を割り当てる
        # これにより、クライアントがこのパスで接続可能になる
        server_sock.bind(QR_SOCKET_PATH)
        
        # ==========================================
        # ステップ5：ソケットファイルのパーミッション設定
        # ==========================================
        # 他のプロセスもこのソケットにアクセスできるように、パーミッションを広くする
        os.chmod(QR_SOCKET_PATH, 0o666)
        
        # ==========================================
        # ステップ6：リッスン開始
        # ==========================================
        # listen() - クライアント接続を待つ準備
        server_sock.listen(1)
        
        print(f"Socket server listening on {QR_SOCKET_PATH}\n")
        
        # ==========================================
        # ステップ7：カメラの初期化
        # ==========================================
        # QRコード読み込み用のカメラを初期化
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            print("Error: Could not open camera", file=sys.stderr)
            server_sock.close()
            return 1
        
        print("Camera initialized successfully\n")
        
        # ==========================================
        # ステップ8：クライアント接続待機
        # ==========================================
        # accept() - ASPプロセスからの接続を待つ
        print("Waiting for client connection...")
        client_sock, _ = server_sock.accept()
        
        print("Client connected")
        
        # ==========================================
        # ステップ9：ハンドシェイク処理
        # ==========================================
        # クライアントからハンドシェイクメッセージを受信
        data = client_sock.recv(BUFFER_SIZE)
        
        if not data:
            print("Error: No handshake data received")
            client_sock.close()
            camera.release()
            server_sock.close()
            return 1
        
        handshake = data.decode('utf-8')
        print(f"Received handshake: {handshake}")
        
        # 期待するメッセージが来たか確認
        if handshake != HANDSHAKE_ACK:
            print("Error: Invalid handshake message")
            client_sock.close()
            camera.release()
            server_sock.close()
            return 1
        
        # ハンドシェイク応答送信
        client_sock.send(HANDSHAKE_MSG.encode('utf-8'))
        print("Handshake successful\n")
        
        # ==========================================
        # ステップ10：メインループ - QRコード読み込み＆送信
        # ==========================================
        # シングルスレッドなので、以下を順序立てて実行：
        # 1. カメラからフレームをキャプチャ
        # 2. QRコードを認識
        # 3. 新しいコードが見つかったらクライアントに送信
        # 4. クライアント切断で終了
        
        print("Starting QR recognition loop...\n")
        
        while running:
            # フレームキャプチャ
            ret, frame = camera.read()
            
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # グレースケール変換（処理を高速化）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # リサイズしてQRコード認識を高速化
            resized = cv2.resize(gray, (640, 480))
            
            # QRコード認識
            barcodes = pyzbar.decode(resized)
            
            # 認識結果を処理
            for barcode in barcodes:
                qr_data = barcode.data.decode('utf-8')
                
                print(f"QR Code detected: {qr_data}")
                
                # 同じデータが連続で来ないようにチェック
                if qr_data != last_qr_data:
                    last_qr_data = qr_data
                    
                    try:
                        # クライアント（ASP）へデータ送信
                        client_sock.send(qr_data.encode('utf-8'))
                        print(f"Sent to client: {qr_data}")
                    except (BrokenPipeError, ConnectionResetError):
                        print("Error: Client disconnected")
                        running = False
                        break
            
            # CPU負荷軽減（100msごとに読み込み）
            time.sleep(0.1)
        
        # ==========================================
        # クリーンアップ
        # ==========================================
        print("\nServer shutting down...")
        client_sock.close()
        camera.release()
        server_sock.close()
        
        if os.path.exists(QR_SOCKET_PATH):
            os.remove(QR_SOCKET_PATH)
        
        print("QR Reader Server stopped")
        return 0
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        server_sock.close()
        if os.path.exists(QR_SOCKET_PATH):
            os.remove(QR_SOCKET_PATH)
        return 1


# ==========================================
# プログラムエントリーポイント
# ==========================================
if __name__ == "__main__":
    sys.exit(main())
