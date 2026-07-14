#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ゲート画像データ収集ツール

説明：
  走行体に搭載したUSBカメラで、カラーゲート（赤・青・黄）を
  様々な角度・距離から撮影し、学習データとして保存するツールです。

  ゲート自体は動かさず、走行体（カメラ）をゲートの周りで
  少しずつ動かしながら撮影することを想定しています。
  （例：ゲートを中心に円を描くように走行体を移動させ、
        各位置で停止して撮影する）

使い方の例：
  # 手動モード：Enterキーを押すたびに1枚撮影する
  # （角度を変えるたびに人が操作するのに向いている）
  python3 capture_gate_images.py --mode manual --label red_gate

  # 自動モード：一定間隔で自動的に撮影し続ける
  # （走行体を一定速度でゆっくり回転させながら撮る場合など）
  python3 capture_gate_images.py --mode auto --interval 1.0 --count 30 --label blue_gate

  # 大量撮影（数百〜数千枚）：Enterキー不要、Ctrl+Cで停止するまで無制限に撮影
  # カメラを持ってゲートの周りをゆっくり歩きながら回るイメージ
  python3 capture_gate_images.py --mode auto --interval 0.2 --count 0 \
      --label red_gate --skip-similar 5

  # カメラ番号を指定したい場合（USBカメラが複数ある場合など）
  python3 capture_gate_images.py --camera 0 --mode manual --label yellow_gate

  # PC側に保存したい場合：PC側で image_receiver.py を先に起動しておき、
  # そのPCのIPアドレスを指定する（Bluetooth PAN経由なら172.16.16.x等）
  python3 capture_gate_images.py --mode manual --label red_gate \
      --save-to remote --remote-host 172.16.16.2

保存構成：
  dataset/
    └ {label}_{セッション日時}/
        ├ {label}_000_{タイムスタンプ}.jpg
        ├ {label}_001_{タイムスタンプ}.jpg
        ├ ...
        └ metadata.csv   （撮影ログ：連番・時刻・角度メモなど）

必要ライブラリ：
  opencv-python（pip install opencv-python）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import socket
import struct
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol


# ==========================================
# カメラの抽象化（実機カメラ／テスト用モックの両方に対応するため）
# ==========================================
class CameraLike(Protocol):
    """
    cv2.VideoCapture と同じインタフェースを持つオブジェクトの型。
    テスト時にモックカメラを差し込めるようにするための定義。
    """

    def read(self):
        ...

    def isOpened(self) -> bool:
        ...

    def release(self) -> None:
        ...


# ==========================================
# 画像の保存先を抽象化（ローカル保存／PCへのネットワーク送信を切り替える）
# ==========================================
class ImageSink(Protocol):
    """
    撮影した画像1枚ぶんを「どこに」保存するかを担当するインタフェース。

    説明：
      LocalFileSink   ：Raspberry Pi内のディスクにそのまま保存する（従来の挙動）
      NetworkImageSink：撮影した画像をその場でPC側へ送信し、PC側で保存する

      CaptureSession/run_captureは、保存先がローカルかリモートかを
      意識せずに同じ呼び出し方（write）で使えるようにするための抽象化。
    """

    def write(
        self, frame, index: int, filename: str, timestamp: str, angle_note: str
    ) -> None:
        ...

    def describe_location(self) -> str:
        ...

    def close(self) -> None:
        ...


def make_session_name(label: str) -> str:
    """
    1回の撮影セッションを識別する名前を生成する。

    説明：
      ローカル保存・ネットワーク送信のどちらであっても、
      Pi側とPC側で同じフォルダ名を使いたいため、
      セッション名の生成をこの1箇所に集約している。

    Returns:
        例："red_gate_20260714_120000"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{label}_{timestamp}"


# ==========================================
# ローカル保存用シンク（従来通り、Raspberry Pi内に保存する）
# ==========================================
class LocalFileSink:
    """Raspberry Pi内のディスクに画像とmetadata.csvを保存するシンク"""

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self._metadata_path = self.session_dir / "metadata.csv"
        self._metadata_file = open(
            self._metadata_path, "w", newline="", encoding="utf-8"
        )
        self._csv_writer = csv.writer(self._metadata_file)
        self._csv_writer.writerow(["index", "filename", "timestamp", "angle_note"])

    def write(
        self, frame, index: int, filename: str, timestamp: str, angle_note: str
    ) -> None:
        import cv2  # 遅延importでテスト容易性を確保

        filepath = self.session_dir / filename
        cv2.imwrite(str(filepath), frame)

        self._csv_writer.writerow([index, filename, timestamp, angle_note])
        self._metadata_file.flush()

    def describe_location(self) -> str:
        return str(self.session_dir)

    def close(self) -> None:
        self._metadata_file.close()


# ==========================================
# ネットワーク送信用シンク（PC側の image_receiver.py へ送信する）
# ==========================================
# 通信プロトコル（image_receiver.py と対になる仕様）：
#   1. TCP接続後、ハンドシェイク
#        送信： b"CAPTURE_CLIENT"
#        期待する応答： b"CAPTURE_SERVER"
#   2. 以降、以下の「フレーム付きメッセージ」を繰り返す
#        [4byte big-endian: JSONヘッダーの長さ][JSONヘッダー(utf-8)]
#        （画像データを伴う場合のみ）
#        [4byte big-endian: 画像データの長さ][JPEGバイト列]
#      JSONヘッダーの type は "session_start" / "image" / "session_end" のいずれか
# ==========================================
CAPTURE_HANDSHAKE_MSG = b"CAPTURE_CLIENT"
CAPTURE_HANDSHAKE_ACK = b"CAPTURE_SERVER"


def _send_framed(sock: socket.socket, data: bytes) -> None:
    """4byteの長さプレフィックス付きでデータを送信する"""
    sock.sendall(struct.pack(">I", len(data)))
    if data:
        sock.sendall(data)


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


class NetworkImageSink:
    """
    撮影した画像を、その場でTCP経由でPC側（image_receiver.py）へ送信し、
    PC側のディスクに保存させるためのシンク。

    説明：
      カメラはRaspberry Pi側にしか物理的に存在しないため撮影自体は
      Pi側で行うが、保存先だけをPC側に変えたい場合に使用する。
    """

    def __init__(self, host: str, port: int, session_name: str,
                 connect_retry: int = 20, retry_interval_sec: float = 0.5,
                 connect_timeout_sec: float = 3.0,
                 recv_timeout_sec: float = 15.0) -> None:
        self.host = host
        self.port = port
        self.session_name = session_name

        sock: socket.socket | None = None
        connected = False
        last_error: Exception | None = None

        for attempt in range(connect_retry):
            print(
                f"[Info] PC側（{host}:{port}）への接続を試みています... "
                f"({attempt + 1}/{connect_retry})"
            )
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # connect()自体にタイムアウトを設けないと、ネットワーク側の
            # トラブル（ファイアウォールでの黙殺など）があった場合に
            # OSレベルの応答待ちで非常に長時間（見た目上「固まった」
            # ように見える状態に）なることがあるため、明示的に設定する。
            sock.settimeout(connect_timeout_sec)
            try:
                sock.connect((host, port))
                connected = True
                break
            except OSError as e:
                last_error = e
                sock.close()
                if attempt < connect_retry - 1:
                    time.sleep(retry_interval_sec)

        if not connected:
            raise ConnectionError(
                f"PC側の受信プログラム（image_receiver.py）に接続できませんでした"
                f"（{host}:{port}）。以下を確認してください。\n"
                f"  ・PC側で image_receiver.py が起動しているか\n"
                f"  ・IPアドレス・ポート番号が正しいか\n"
                f"  ・PC側のファイアウォールが該当ポートの着信を"
                f"許可しているか（Windowsの場合、初回起動時に表示される"
                f"許可ダイアログでブロックしたままになっていないか）\n"
                f"詳細: {last_error}"
            )

        assert sock is not None
        # 接続後は、ハンドシェイクや画像転送時に相手からの応答が
        # 完全に途絶えた場合でも無限に待ち続けないよう、
        # 有限のタイムアウトを設定しておく（真の意味でのフリーズを防ぐ）。
        sock.settimeout(recv_timeout_sec)

        # ハンドシェイク
        try:
            sock.sendall(CAPTURE_HANDSHAKE_MSG)
            response = sock.recv(len(CAPTURE_HANDSHAKE_ACK))
        except socket.timeout as e:
            sock.close()
            raise ConnectionError(
                f"ハンドシェイクの応答がタイムアウトしました（{recv_timeout_sec}秒）。"
                f"接続は確立できましたが、PC側から応答がありません。"
                f"ファイアウォールが接続確立後の通信を妨げていないか、"
                f"image_receiver.py が正しく動作しているか確認してください。"
            ) from e

        if response != CAPTURE_HANDSHAKE_ACK:
            sock.close()
            raise ConnectionError(
                f"ハンドシェイクに失敗しました（応答: {response!r}）。"
                f"接続先が image_receiver.py であるか確認してください。"
            )

        print("[Info] PC側とのハンドシェイクに成功しました。")

        self._sock = sock

        # セッション開始をPC側に通知（フォルダ名を揃えるため）
        header = json.dumps(
            {"type": "session_start", "session_name": session_name}
        ).encode("utf-8")
        _send_framed(self._sock, header)

    def write(
        self, frame, index: int, filename: str, timestamp: str, angle_note: str
    ) -> None:
        import cv2

        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("画像のJPEGエンコードに失敗しました")
        img_bytes = encoded.tobytes()

        header = json.dumps(
            {
                "type": "image",
                "index": index,
                "filename": filename,
                "timestamp": timestamp,
                "angle_note": angle_note,
                "size": len(img_bytes),
            }
        ).encode("utf-8")

        _send_framed(self._sock, header)
        _send_framed(self._sock, img_bytes)

        # PC側からの受信確認（1byte）を待つことで、送信の詰まりや
        # 切断を早期に検知できるようにする
        try:
            ack = self._sock.recv(1)
        except socket.timeout as e:
            raise ConnectionError(
                "PC側からの受信確認がタイムアウトしました。"
                "通信が途中で止まっている可能性があります。"
            ) from e

        if ack != b"K":
            raise ConnectionError(
                f"PC側からの受信確認が得られませんでした（応答: {ack!r}）"
            )

    def describe_location(self) -> str:
        return f"{self.host}:{self.port} （PC側、セッション名: {self.session_name}）"

    def close(self) -> None:
        try:
            header = json.dumps({"type": "session_end"}).encode("utf-8")
            _send_framed(self._sock, header)
        finally:
            self._sock.close()


# ==========================================
# 実行フラグ（Ctrl+Cでの安全な終了用）
# ==========================================
_running = True


def _signal_handler(sig, frame):
    global _running
    print("\n[Info] 終了シグナルを受信しました。安全に終了します...")
    _running = False
    # 単に_runningをFalseにしてreturnするだけだと、PEP 475の仕様により
    # input()などブロッキング中の処理が「割り込まれなかったこと」にされ、
    # 自動的に再開してしまう（＝Enterキーをもう一度押すまで終了しない）。
    # 明示的に例外を送出することで、input()等の待機を確実に中断させる。
    raise KeyboardInterrupt


# ==========================================
# 設定
# ==========================================
@dataclass
class CaptureConfig:
    label: str                     # 撮影対象のラベル（例："red_gate"）
    mode: str = "manual"           # "manual" または "auto"
    camera_index: int = 0          # カメラ番号
    interval_sec: float = 1.0      # autoモードでの撮影間隔（秒）
    count: int = 20                # 撮影する総枚数（0以下を指定すると無制限：Ctrl+Cで終了）
    outdir: Path = field(default_factory=lambda: Path("dataset"))
    resize_width: int | None = None   # 指定時はリサイズして保存（例：640）
    ask_angle: bool = False        # manualモードで角度メモを毎回入力するか
    position_plan: list[str] | None = None  # 撮影位置の指示リスト（自動生成 or 手動指定）
    countdown_sec: int = 3         # autoモード開始前のカウントダウン秒数
    skip_similar_threshold: float = 0.0
    # ↑ 0より大きい値を指定すると、直前の保存フレームとの差分が
    #   この値未満の場合は保存をスキップする（立ち止まっている間の
    #   ほぼ同一画像の量産を防ぐ）。差分は0〜255スケールの平均絶対差。
    #   目安：3〜8程度で「ほぼ静止」を検出しやすい。
    max_consecutive_non_saves: int = 3000
    # ↑ 「保存されない」状態（フレーム取得失敗、または類似フレームで
    #   スキップ）が連続でこの回数を超えたら、ハングを避けるため
    #   安全に撮影を打ち切る。カメラが完全に壊れた場合や、
    #   被写体が全く動かない状況が続いた場合の保険。
    warmup_frames: int = 5
    # ↑ カメラを開いた直後は、自動露出・自動ホワイトバランスが
    #   安定するまで数フレームかかることが多い。特に本ツールは
    #   「色」を正確に捉えることが目的なので、安定前のフレームが
    #   学習データに混ざらないよう、最初にこの枚数だけ読み捨てる。
    #   （qr_reader.py はQRの形状しか見ないためこの配慮は不要だが、
    #     色ゲート検出用データではホワイトバランスのブレが致命的になりうる）
    frame_width: int | None = None
    frame_height: int | None = None
    # ↑ 指定するとカメラの解像度をこの値に固定する。将来、色検出用の
    #   画像処理を作る際に学習データと実行時の解像度がズレないようにする。
    pre_capture_flush_frames: int = 3
    # ↑ 実際に保存する直前に、この枚数分だけ余分にフレームを読み捨てる。
    #   カメラ内部バッファに溜まった古いフレームを吐き出し、
    #   「今」のフレームを確実に取得するための対策。
    #   特にmanualモード（Enterキー待ちで間隔が空く）で重要。
    #   0にすると無効化（バッファ問題がない環境向け）。
    save_to: str = "local"          # "local"（Pi内保存）または "remote"（PC側へ送信）
    remote_host: str | None = None  # save_to="remote"時の接続先（PC側のIPアドレス）
    remote_port: int = 9999         # save_to="remote"時の接続先ポート
    remote_connect_retry: int = 20
    remote_retry_interval_sec: float = 0.5


# ==========================================
# 撮影位置プランの自動生成
# ==========================================
def build_position_plan(
    num_directions: int = 8,
    distances: tuple[str, ...] = ("near", "mid", "far"),
) -> list[str]:
    """
    「ゲートの周りを人が回りながら撮る」ことを想定し、
    網羅的な撮影位置のリストを自動生成する。

    説明：
      num_directions=8 なら、ゲートを中心に45度刻みで8方向。
      distances=("near","mid","far") なら、各方向で3段階の距離。
      合計 8 × 3 = 24 枚分の位置ラベルが生成される。

      生成される順番は「同じ距離を1周してから次の距離へ」なので、
      実際に人がゲートの周りを1周ずつ回る動線と一致する。

    Args:
        num_directions: 方向の分割数（8なら45度刻み、4なら90度刻み）
        distances:      距離のバリエーション（近い順で書くことを推奨）

    Returns:
        位置ラベルのリスト（例：["0deg_near", "45deg_near", ...]）
    """
    if num_directions < 1:
        raise ValueError("num_directions は1以上である必要があります")

    step_deg = 360 // num_directions
    plan: list[str] = []

    for distance in distances:
        for i in range(num_directions):
            angle = step_deg * i
            plan.append(f"{angle}deg_{distance}")

    return plan


# ==========================================
# 類似フレーム判定（立ち止まっている間の無駄撮り防止）
# ==========================================
def frame_difference_score(frame_a, frame_b) -> float:
    """
    2つのフレームがどれだけ違うかを、簡易な指標で返す。

    説明：
      グレースケール化＋縮小してから平均絶対差を取ることで、
      画質のわずかなノイズには反応しにくく、かつ高速に計算できる
      ようにしている。厳密な画像類似度ではなく、あくまで
      「ほぼ静止しているかどうか」を判定するための簡易指標。

    Args:
        frame_a, frame_b: 比較する2つのフレーム（numpy配列、BGR想定）

    Returns:
        差分スコア（0〜255程度のスケール。0に近いほど似ている）
    """
    import cv2
    import numpy as np

    small_a = cv2.resize(cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY), (64, 48))
    small_b = cv2.resize(cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY), (64, 48))

    diff = cv2.absdiff(small_a, small_b)
    return float(np.mean(diff))


# ==========================================
# バッファ吐き出し（V4L2などの遅延フレーム対策）
# ==========================================
def read_fresh_frame(cam: CameraLike, flush_count: int = 3):
    """
    カメラ内部にバッファされた「古いフレーム」を読み捨ててから、
    最新のフレームを取得する。

    説明：
      cv2.VideoCapture（特にV4L2バックエンド）は、呼び出し間隔が
      空くと内部バッファに数フレーム分が溜まってしまうことがある。
      その状態でread()を1回呼んでも、返ってくるのは「今」ではなく
      「数フレーム前」の画像であることが多く、
      manualモードのようにEnterキー待ちで間隔が空く撮影方式では
      「毎回同じような画像しか撮れない」という症状として現れる。

      本関数は、flush_count回だけ余分にread()して結果を捨てることで
      バッファを吐き出し、最後にもう一度read()した「本当に新しい」
      フレームを返す。

    Args:
        cam:         カメラオブジェクト
        flush_count: 読み捨てる回数（大きいほど確実だが、その分遅くなる）

    Returns:
        (ok, frame) のタプル。cv2.VideoCapture.read()と同じ形式。
    """
    for _ in range(flush_count):
        cam.read()  # 結果は使わず捨てる（バッファの吐き出し）

    return cam.read()


# ==========================================
# セッション（1回の撮影作業）の管理
# ==========================================
class CaptureSession:
    """
    1回分の撮影セッションを管理するクラス。

    説明：
      実際の保存処理はImageSink（LocalFileSink または NetworkImageSink）に
      委譲し、このクラスは「連番の管理」「ファイル名の組み立て」
      「リサイズ」など、保存先によらず共通の処理に専念する。
    """

    def __init__(
        self, config: CaptureConfig, sink: ImageSink, session_name: str
    ) -> None:
        self.config = config
        self.sink = sink
        self.session_name = session_name
        self._count_saved = 0

    @property
    def session_dir(self) -> Path | None:
        """
        ローカル保存（LocalFileSink）の場合のみ、保存先ディレクトリを返す。
        ネットワーク送信（NetworkImageSink）の場合はNoneを返す。
        （既存のテストコードが session.session_dir を参照しているため、
          ローカル保存時は従来通り動作するように維持している）
        """
        return getattr(self.sink, "session_dir", None)

    def save_frame(self, frame, angle_note: str = "") -> str:
        """
        1枚の画像をシンクへ保存し、metadata相当の情報も記録させる。

        Args:
            frame:      OpenCVのフレーム（numpy配列）
            angle_note: 角度・向きなどの自由記述メモ（任意）

        Returns:
            保存したファイル名（パスではなく名前のみ。
            ネットワーク保存の場合はPi側にファイルの実体がないため）
        """
        import cv2  # 遅延importでテスト容易性を確保

        if self.config.resize_width is not None:
            h, w = frame.shape[:2]
            new_w = self.config.resize_width
            new_h = int(h * (new_w / w))
            frame = cv2.resize(frame, (new_w, new_h))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{self.config.label}_{self._count_saved:03d}_{timestamp}.jpg"

        self.sink.write(frame, self._count_saved, filename, timestamp, angle_note)

        self._count_saved += 1
        return filename

    @property
    def saved_count(self) -> int:
        return self._count_saved

    def close(self) -> None:
        self.sink.close()


# ==========================================
# 撮影ロジック本体（カメラを引数で受け取るのでテスト可能）
# ==========================================
def run_capture(cam: CameraLike, config: CaptureConfig) -> CaptureSession:
    """
    設定に従って撮影を実行する。

    説明：
      manualモード：
        position_planが指定されていれば、その位置ラベルを順番に
        提示しながら1枚ずつ撮影する。
        position_planがなければ、単にEnterキー入力を待つだけの
        シンプルな撮影になる。

      autoモード：
        Enterキー入力は一切不要。カウントダウン後、指定間隔で
        自動的に撮影し続ける。count<=0を指定すると無制限撮影となり、
        Ctrl+Cを押すまで撮り続ける（数百〜数千枚集めたい場合向け）。
        skip_similar_thresholdを指定すると、直前の保存フレームと
        ほぼ同じ画像（＝立ち止まっている間の無駄撮り）を自動でスキップする。

    Args:
        cam:    カメラオブジェクト（cv2.VideoCaptureまたは互換モック）
        config: 撮影設定

    Returns:
        撮影結果を保持した CaptureSession
    """
    global _running
    _running = True

    # position_planが指定されている場合は、枚数をプランの長さに合わせる
    if config.position_plan is not None:
        config.count = len(config.position_plan)

    unlimited = config.count <= 0

    session_name = make_session_name(config.label)
    try:
        if config.save_to == "remote":
            if not config.remote_host:
                raise ValueError(
                    "save_to='remote' の場合は remote_host の指定が必要です"
                )
            sink: ImageSink = NetworkImageSink(
                config.remote_host, config.remote_port, session_name,
                connect_retry=config.remote_connect_retry,
                retry_interval_sec=config.remote_retry_interval_sec,
            )
        else:
            sink = LocalFileSink(config.outdir / session_name)
    except KeyboardInterrupt:
        # PC側への接続待ちなど、撮影ループに入る前の段階でCtrl+Cが
        # 押された場合はここで捕まえ、クラッシュせず穏やかに終了する。
        print("[Info] 接続処理中に中断されました。撮影は開始されていません。")
        raise SystemExit(0)

    session = CaptureSession(config, sink, session_name)

    print(f"[Info] 保存先: {sink.describe_location()}")
    if unlimited:
        print(f"[Info] モード: {config.mode} / 目標枚数: 無制限（Ctrl+Cで終了）\n")
    else:
        print(f"[Info] モード: {config.mode} / 目標枚数: {config.count}\n")

    # カメラの自動露出・自動ホワイトバランスが安定するまで、
    # 最初の数フレームは読み捨てる（色データ収集の精度確保のため）
    if config.warmup_frames > 0:
        print(f"[Info] カメラの自動調整待ち（{config.warmup_frames}フレーム読み捨て中）...")
        for _ in range(config.warmup_frames):
            cam.read()
        print("[Info] カメラの準備が整いました。\n")

    # autoモードは開始前にカウントダウンし、カメラを構える時間を確保する
    if config.mode == "auto" and config.countdown_sec > 0:
        print("[Info] 撮影を開始します。カメラをゲートに向けてください。")
        for remaining in range(config.countdown_sec, 0, -1):
            print(f"  {remaining}...")
            time.sleep(1.0)
        print("[Info] 撮影開始！\n")

    last_saved_frame = None  # 類似フレーム判定用に直前の保存フレームを保持
    non_save_streak = 0      # 連続で「保存されなかった」回数（失敗+スキップ）

    try:
        while _running and (unlimited or session.saved_count < config.count):
            if config.mode == "manual":
                if config.position_plan is not None:
                    current_label = config.position_plan[session.saved_count]
                    prompt = (
                        f"[{session.saved_count + 1}/{config.count}] "
                        f"次の位置 → 「{current_label}」"
                        f"にカメラを構えてEnterキーを押してください..."
                    )
                    input(prompt)
                    angle_note = current_label

                elif config.ask_angle:
                    angle_note = input(
                        f"[{session.saved_count + 1}/{config.count}] "
                        f"角度・メモを入力してEnter（空でも可）: "
                    )
                else:
                    angle_note = ""
                    input(
                        f"[{session.saved_count + 1}/{config.count}] "
                        f"撮影位置に移動したらEnterキーを押してください..."
                    )

                ok, frame = read_fresh_frame(cam, config.pre_capture_flush_frames)
                if not ok:
                    print("[Warn] フレームの取得に失敗しました。リトライします。")
                    continue

                filename = session.save_frame(frame, angle_note)
                print(f"  → 保存しました: {filename}")

            elif config.mode == "auto":
                ok, frame = read_fresh_frame(cam, config.pre_capture_flush_frames)
                if not ok:
                    print("[Warn] フレームの取得に失敗しました。リトライします。")
                    non_save_streak += 1
                    if non_save_streak >= config.max_consecutive_non_saves:
                        print(
                            f"[Error] フレーム取得の失敗が"
                            f"{non_save_streak}回連続しました。"
                            f"カメラの状態を確認してください。安全のため撮影を終了します。"
                        )
                        break
                    time.sleep(config.interval_sec)
                    continue

                # 類似フレームのスキップ判定
                if (
                    config.skip_similar_threshold > 0
                    and last_saved_frame is not None
                ):
                    diff = frame_difference_score(frame, last_saved_frame)
                    if diff < config.skip_similar_threshold:
                        # ほぼ同じ画像なので保存せず次のループへ
                        non_save_streak += 1
                        if non_save_streak >= config.max_consecutive_non_saves:
                            print(
                                f"[Warn] 変化のないフレームが"
                                f"{non_save_streak}回連続しました"
                                f"（被写体が動いていない可能性があります）。"
                                f"安全のため撮影を終了します。"
                            )
                            break
                        time.sleep(config.interval_sec)
                        continue

                filename = session.save_frame(frame)
                last_saved_frame = frame
                non_save_streak = 0  # 保存できたのでリセット

                count_display = (
                    f"{session.saved_count}"
                    if unlimited
                    else f"{session.saved_count}/{config.count}"
                )
                print(f"[{count_display}] 保存しました: {filename}")
                time.sleep(config.interval_sec)

            else:
                raise ValueError(f"不明なモードです: {config.mode}")

    except KeyboardInterrupt:
        # Ctrl+C（SIGINT）による割り込み。_signal_handlerが既に
        # _running=Falseを設定し、終了メッセージも表示済みなので、
        # ここでは何もせず正常にループを抜けて後続の集計表示へ進む。
        pass

    finally:
        session.close()

    print(f"\n[Info] 撮影終了。合計 {session.saved_count} 枚を保存しました。")
    if session.session_dir is not None:
        print(f"[Info] 保存先フォルダ: {session.session_dir}")
        print(f"[Info] メタデータ: {session.session_dir / 'metadata.csv'}")
    else:
        print(f"[Info] 保存先: {sink.describe_location()}")
        print(f"[Info] （PC側でファイルとmetadata.csvが作成されています）")

    return session


# ==========================================
# 実カメラを使ったエントリポイント
# ==========================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="ゲート画像データ収集ツール"
    )
    parser.add_argument(
        "--label", required=True,
        help="撮影対象のラベル（例: red_gate, blue_gate, yellow_gate）"
    )
    parser.add_argument(
        "--mode", choices=["manual", "auto"], default="manual",
        help="manual: Enterキーで1枚ずつ撮影 / auto: 一定間隔で自動撮影"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="使用するカメラ番号（デフォルト: 0）"
    )
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="autoモードでの撮影間隔（秒、デフォルト: 1.0）"
    )
    parser.add_argument(
        "--count", type=int, default=20,
        help="撮影する総枚数（デフォルト: 20）。0以下を指定すると無制限"
             "（Ctrl+Cを押すまで撮り続ける）。数百〜数千枚集めたい場合に便利"
    )
    parser.add_argument(
        "--outdir", type=str, default="dataset",
        help="保存先のルートフォルダ（デフォルト: dataset）"
    )
    parser.add_argument(
        "--resize-width", type=int, default=None,
        help="指定時はこの幅にリサイズして保存する（アスペクト比維持）"
    )
    parser.add_argument(
        "--ask-angle", action="store_true",
        help="manualモード時、撮影ごとに角度メモの入力を求める"
    )
    parser.add_argument(
        "--directions", type=int, default=None,
        help="撮影プランを自動生成：方向の分割数（例: 8なら45度刻み）。"
             "指定するとpositionプランに従った案内が表示される"
    )
    parser.add_argument(
        "--distances", type=str, default="near,mid,far",
        help="撮影プランで使う距離ラベル（カンマ区切り、デフォルト: near,mid,far）"
    )
    parser.add_argument(
        "--countdown", type=int, default=3,
        help="autoモード開始前のカウントダウン秒数（デフォルト: 3、0で即開始）"
    )
    parser.add_argument(
        "--skip-similar", type=float, default=0.0,
        help="autoモードで、直前の保存フレームとの差分がこの値未満なら"
             "保存をスキップする（立ち止まっている間の無駄撮り防止）。"
             "0で無効（デフォルト）。目安は3〜8程度"
    )
    parser.add_argument(
        "--max-non-saves", type=int, default=3000,
        help="連続で保存されない状態（取得失敗・類似スキップ）が"
             "この回数を超えたら安全のため撮影を打ち切る（デフォルト: 3000）"
    )
    parser.add_argument(
        "--warmup-frames", type=int, default=5,
        help="撮影開始前に読み捨てるフレーム数。カメラの自動露出・"
             "自動ホワイトバランスが安定するのを待つため（デフォルト: 5）"
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="カメラの解像度（幅）を固定したい場合に指定する"
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help="カメラの解像度（高さ）を固定したい場合に指定する"
    )
    parser.add_argument(
        "--pre-capture-flush", type=int, default=3,
        help="保存直前に読み捨てるフレーム数。カメラ内部バッファに溜まった"
             "古いフレームを吐き出し、複数枚撮っても同じ画像しか保存されない"
             "現象を防ぐ（デフォルト: 3）"
    )
    parser.add_argument(
        "--save-to", choices=["local", "remote"], default="local",
        help="local: Raspberry Pi内に保存（デフォルト） / "
             "remote: PC側（image_receiver.py）へ送信して保存させる"
    )
    parser.add_argument(
        "--remote-host", type=str, default=None,
        help="--save-to remote 時の接続先PCのIPアドレス（例: 172.16.16.2）"
    )
    parser.add_argument(
        "--remote-port", type=int, default=9999,
        help="--save-to remote 時の接続先ポート番号（デフォルト: 9999）"
    )

    args = parser.parse_args()

    position_plan = None
    if args.directions is not None:
        distance_list = tuple(
            d.strip() for d in args.distances.split(",") if d.strip()
        )
        position_plan = build_position_plan(
            num_directions=args.directions,
            distances=distance_list,
        )

    config = CaptureConfig(
        label=args.label,
        mode=args.mode,
        camera_index=args.camera,
        interval_sec=args.interval,
        count=args.count,
        outdir=Path(args.outdir),
        resize_width=args.resize_width,
        ask_angle=args.ask_angle,
        position_plan=position_plan,
        countdown_sec=args.countdown,
        skip_similar_threshold=args.skip_similar,
        max_consecutive_non_saves=args.max_non_saves,
        warmup_frames=args.warmup_frames,
        frame_width=args.width,
        frame_height=args.height,
        pre_capture_flush_frames=args.pre_capture_flush,
        save_to=args.save_to,
        remote_host=args.remote_host,
        remote_port=args.remote_port,
    )

    signal.signal(signal.SIGINT, _signal_handler)

    import cv2

    cam = cv2.VideoCapture(config.camera_index)

    if config.frame_width is not None:
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    if config.frame_height is not None:
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)

    # バッファを可能な限り小さくし、古いフレームが溜まりにくくする
    # （対応していない環境では無視されるだけで害はない）
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cam.isOpened():
        print(
            f"[Error] カメラ {config.camera_index} を開けませんでした。",
            file=sys.stderr,
        )
        print(
            "[Hint] qr_reader.py など、同じカメラを既に使用している"
            "プロセスが起動していないか確認してください。"
            "USBカメラは基本的に1プロセスからしか同時に使用できません。"
            f"（例: ps aux | grep qr_reader、"
            f"lsof /dev/video{config.camera_index} などで確認できます）",
            file=sys.stderr,
        )
        return 1

    try:
        run_capture(cam, config)
    finally:
        cam.release()

    return 0


if __name__ == "__main__":
    sys.exit(main())
