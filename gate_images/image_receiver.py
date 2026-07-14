#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ゲート画像 受信サーバー（PC側）

説明：
  capture_gate_images.py（Raspberry Pi側）から --save-to remote で
  送信されてきた画像を受信し、PC側のディスクに保存するプログラムです。

  カメラは走行体（Raspberry Pi）にしか物理的に無いため撮影はPi側で行い、
  保存先だけをこのPC側に変えたい、という用途で使います。

通信プロトコル（capture_gate_images.py 側と対）：
  1. TCP接続後、ハンドシェイク
       受信： b"CAPTURE_CLIENT"
       応答： b"CAPTURE_SERVER"
  2. 以降、次の形式のメッセージを繰り返し受信する
       [4byte big-endian: JSONヘッダーの長さ][JSONヘッダー(utf-8)]
       （画像データを伴う場合のみ）
       [4byte big-endian: 画像データの長さ][JPEGバイト列]
     JSONヘッダーの type は "session_start" / "image" / "session_end"

使い方：
  # PC側でまずこちらを起動して待ち受ける
  python3 image_receiver.py --outdir dataset --port 9999

  # その後、Raspberry Pi側で以下のように実行する
  # python3 capture_gate_images.py --mode manual --label red_gate \
  #     --save-to remote --remote-host <このPCのIPアドレス>

必要ライブラリ：
  opencv-python（pip install opencv-python）
  ※画像そのものはJPEGバイト列として届くため、本来はデコード不要で
    そのままファイルに書き出せるが、簡単な健全性チェック
    （壊れたJPEGでないかの確認）のためにOpenCVで検証用途に使用する。
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import socket
import struct
import sys
from pathlib import Path


CAPTURE_HANDSHAKE_MSG = b"CAPTURE_CLIENT"
CAPTURE_HANDSHAKE_ACK = b"CAPTURE_SERVER"
BACKLOG = 1

_running = True


def _signal_handler(sig, frame):
    global _running
    print("\n[Info] 終了シグナルを受信しました。サーバーを停止します...")
    _running = False
    raise KeyboardInterrupt


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    """指定バイト数を確実に受信する（部分受信を繰り返し結合する）"""
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("接続が途中で切断されました")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_framed(sock: socket.socket) -> bytes:
    """4byteの長さプレフィックス付きのデータを受信する"""
    (length,) = struct.unpack(">I", _recv_exact(sock, 4))
    if length == 0:
        return b""
    return _recv_exact(sock, length)


class SessionWriter:
    """
    1回の撮影セッション（1接続）ぶんの画像・metadata.csvを
    ディスクへ書き出すクラス。Pi側のLocalFileSinkとほぼ同じ役割を、
    受信したデータに対して行う。
    """

    def __init__(self, outdir: Path, session_name: str) -> None:
        self.session_dir = outdir / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self._metadata_path = self.session_dir / "metadata.csv"
        self._metadata_file = open(
            self._metadata_path, "w", newline="", encoding="utf-8"
        )
        self._csv_writer = csv.writer(self._metadata_file)
        self._csv_writer.writerow(["index", "filename", "timestamp", "angle_note"])
        self._saved_count = 0

    def save_image(
        self, filename: str, timestamp: str, angle_note: str, img_bytes: bytes
    ) -> Path:
        filepath = self.session_dir / filename
        with open(filepath, "wb") as f:
            f.write(img_bytes)

        self._csv_writer.writerow(
            [self._saved_count, filename, timestamp, angle_note]
        )
        self._metadata_file.flush()
        self._saved_count += 1
        return filepath

    @property
    def saved_count(self) -> int:
        return self._saved_count

    def close(self) -> None:
        self._metadata_file.close()


def handle_connection(conn: socket.socket, outdir: Path) -> None:
    """
    1つのクライアント接続を最後まで処理する。

    説明：
      ハンドシェイク → session_start → image を繰り返し受信 → session_end
      という一連の流れを1つの接続の中で処理する。

    Args:
        conn:   接続済みのソケット
        outdir: 保存先のルートフォルダ
    """
    # ハンドシェイク
    data = conn.recv(len(CAPTURE_HANDSHAKE_MSG))
    if data != CAPTURE_HANDSHAKE_MSG:
        print(f"[Warn] 不正なハンドシェイクです: {data!r}")
        return
    conn.sendall(CAPTURE_HANDSHAKE_ACK)
    print("[Info] クライアントとのハンドシェイクに成功しました")

    writer: SessionWriter | None = None

    try:
        while True:
            header_bytes = _recv_framed(conn)
            header = json.loads(header_bytes.decode("utf-8"))
            msg_type = header.get("type")

            if msg_type == "session_start":
                session_name = header["session_name"]
                writer = SessionWriter(outdir, session_name)
                print(f"[Info] セッション開始: {writer.session_dir}")

            elif msg_type == "image":
                if writer is None:
                    print("[Warn] session_start前にimageを受信しました。無視します。")
                    continue

                img_bytes = _recv_framed(conn)
                filepath = writer.save_image(
                    filename=header["filename"],
                    timestamp=header["timestamp"],
                    angle_note=header.get("angle_note", ""),
                    img_bytes=img_bytes,
                )
                print(f"  [{writer.saved_count}] 保存しました: {filepath.name}")

                # Pi側へ受信確認を返す（送信側の詰まり検知のため）
                conn.sendall(b"K")

            elif msg_type == "session_end":
                print(
                    f"[Info] セッション終了: 合計 "
                    f"{writer.saved_count if writer else 0} 枚を保存しました"
                )
                break

            else:
                print(f"[Warn] 不明なメッセージタイプです: {msg_type}")

    except ConnectionError as e:
        print(f"[Warn] 接続が切断されました: {e}")

    finally:
        if writer is not None:
            writer.close()


def run_server(outdir: Path, port: int, host: str = "0.0.0.0") -> None:
    """
    サーバーを起動し、接続を待ち受け続ける。

    説明：
      capture_gate_images.py 側は1回の撮影セッションにつき1回接続してくる
      想定のため、1接続を最後まで処理したら、また次の接続を待つ
      ループ構成にしている（シングルスレッドで十分な用途のため）。
    """
    global _running
    _running = True

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(BACKLOG)

    print(f"[Info] 画像受信サーバーを起動しました: {host}:{port}")
    print(f"[Info] 保存先ルートフォルダ: {outdir.resolve()}")
    print("[Info] Raspberry Pi側からの接続を待っています...\n")

    try:
        while _running:
            server_sock.settimeout(1.0)
            try:
                conn, addr = server_sock.accept()
            except socket.timeout:
                continue

            print(f"[Info] 接続を受け付けました: {addr}")
            try:
                handle_connection(conn, outdir)
            finally:
                conn.close()
            print("[Info] 次の接続を待っています...\n")

    except KeyboardInterrupt:
        pass

    finally:
        server_sock.close()
        print("[Info] サーバーを停止しました")


def main() -> int:
    parser = argparse.ArgumentParser(description="ゲート画像 受信サーバー（PC側）")
    parser.add_argument(
        "--outdir", type=str, default="dataset",
        help="保存先のルートフォルダ（デフォルト: dataset）"
    )
    parser.add_argument(
        "--port", type=int, default=9999,
        help="待ち受けるポート番号（デフォルト: 9999、Pi側の--remote-portと合わせること）"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="待ち受けるアドレス（デフォルト: 0.0.0.0＝すべてのネットワークから受付）"
    )

    args = parser.parse_args()

    signal.signal(signal.SIGINT, _signal_handler)

    run_server(Path(args.outdir), args.port, args.host)
    return 0


if __name__ == "__main__":
    sys.exit(main())
