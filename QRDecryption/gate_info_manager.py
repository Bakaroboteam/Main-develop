#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ゲート位置情報 復号モジュール（統合版）

説明：
  ETロボコン アプライドクラスの「ヒントカード」に記録された
  ゲート位置情報を取得・復号するモジュールです。

  以下の3つの機能を提供します：
    1. 復号キーの入力機能（キャリブレーション時にスターターが入力）
    2. 入力した復号キーを、QRから暗号文を取得するまで保持する機能
    3. qr_reader.py（QRコード認識サーバー）のソケットから
       QRコードの中身（暗号文/平文）を受け取る機能

  規約7.3.1節の仕様：
    暗号化方式　　：AES-128(ECB)
    暗号文出力形式：Base64
    復号キー形式　：テキスト（4桁）
"""

from __future__ import annotations

import base64
import socket
from dataclasses import dataclass, field
from enum import Enum, auto

from Crypto.Cipher import AES


QR_SOCKET_PATH = "/tmp/qr_socket"
HANDSHAKE_MSG = "ASP_QR_CLIENT"
HANDSHAKE_ACK = "QR_SERVER"
BUFFER_SIZE = 1024


class GateDecodeError(Exception):
    pass


class DecryptionKeyError(Exception):
    pass


class QrConnectionError(Exception):
    pass


@dataclass(frozen=True)
class GatePosition:
    start: tuple[int, int]
    end: tuple[int, int]

    def __repr__(self) -> str:
        sx, sy = self.start
        ex, ey = self.end
        return f"GatePosition(G{sx}-{sy} - G{ex}-{ey})"


class HintCardKind(Enum):
    HINT_CARD_1 = auto()
    HINT_CARD_2 = auto()
    POSITION_ASSIST = auto()
    UNKNOWN = auto()


# ==========================================
# 1. 復号キーの入力・保持機能
# ==========================================
class DecryptionKeyStore:
    """
    復号キーを入力・保持するクラス。
    規約7.2.1節「キャリブレーション中の復号キー入力」に対応。
    """

    def __init__(self) -> None:
        self._key: str | None = None

    def set_key(self, decryption_key: str) -> None:
        if not (decryption_key.isdigit() and len(decryption_key) == 4):
            raise DecryptionKeyError(
                f"復号キーは4桁の数字である必要があります: {decryption_key!r}"
            )
        self._key = decryption_key
        print(f"[KeyStore] 復号キーを設定しました: {decryption_key}")

    def get_key(self) -> str:
        if self._key is None:
            raise DecryptionKeyError(
                "復号キーがまだ設定されていません。set_key() を先に呼び出してください。"
            )
        return self._key

    def has_key(self) -> bool:
        return self._key is not None

    def clear(self) -> None:
        self._key = None
        print("[KeyStore] 復号キーをクリアしました")


def _normalize_key(decryption_key: str) -> bytes:
    """
    4桁の復号キーをAES-128用16byte鍵に変換する。
    ※鍵生成方式は規約に明記がないため、実行委員会配布の
      暗号化ロジックと必ず照合すること。
    """
    if not (decryption_key.isdigit() and len(decryption_key) == 4):
        raise GateDecodeError(
            f"復号キーは4桁の数字である必要があります: {decryption_key!r}"
        )
    key_bytes = decryption_key.encode("utf-8")
    return key_bytes.ljust(16, b"\x00")


def _pkcs7_unpad(data: bytes) -> bytes:
    if not data:
        raise GateDecodeError("復号結果が空です")
    pad_len = data[-1]
    if pad_len < 1 or pad_len > AES.block_size:
        raise GateDecodeError(f"不正なパディング長です: {pad_len}")
    if data[-pad_len:] != bytes([pad_len]) * pad_len:
        raise GateDecodeError("パディングの内容が不正です（鍵が違う可能性があります）")
    return data[:-pad_len]


def decrypt_gate_info(cipher_text_b64: str, decryption_key: str) -> str:
    key = _normalize_key(decryption_key)
    try:
        cipher_bytes = base64.b64decode(cipher_text_b64, validate=True)
    except Exception as e:
        raise GateDecodeError(f"Base64デコードに失敗しました: {e}") from e

    if len(cipher_bytes) == 0 or len(cipher_bytes) % AES.block_size != 0:
        raise GateDecodeError(
            f"暗号文の長さが不正です（16byteの倍数である必要があります）: {len(cipher_bytes)}byte"
        )

    cipher = AES.new(key, AES.MODE_ECB)
    decrypted_padded = cipher.decrypt(cipher_bytes)
    decrypted_bytes = _pkcs7_unpad(decrypted_padded)

    try:
        return decrypted_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise GateDecodeError(
            "復号結果がUTF-8として解釈できません（鍵が違う可能性があります）"
        ) from e


def parse_gate_positions(plain_text: str) -> list[GatePosition]:
    plain_text = plain_text.strip()
    if not plain_text:
        raise GateDecodeError("座標文字列が空です")

    gate_positions: list[GatePosition] = []

    for gate_chunk in plain_text.split("/"):
        gate_chunk = gate_chunk.strip()
        points_raw = gate_chunk.split(",")
        if len(points_raw) != 2:
            raise GateDecodeError(
                f"ゲート位置の形式が不正です（'XY,XY'である必要があります）: {gate_chunk!r}"
            )
        points: list[tuple[int, int]] = []
        for point_str in points_raw:
            point_str = point_str.strip()
            if len(point_str) != 2 or not point_str.isdigit():
                raise GateDecodeError(
                    f"座標の形式が不正です（2桁の数字である必要があります）: {point_str!r}"
                )
            x = int(point_str[0])
            y = int(point_str[1])
            if not (1 <= x <= 5 and 1 <= y <= 5):
                raise GateDecodeError(
                    f"座標がゲートポジションの範囲(1-5)を超えています: G{x}-{y}"
                )
            points.append((x, y))
        gate_positions.append(GatePosition(start=points[0], end=points[1]))

    return gate_positions


def parse_hint_card_1(plain_text: str) -> GatePosition:
    positions = parse_gate_positions(plain_text)
    if len(positions) != 1:
        raise GateDecodeError(
            f"ヒントカード1は1ゲート分の情報である必要があります: {plain_text!r}"
        )
    return positions[0]


def classify_qr_text(qr_text: str) -> HintCardKind:
    """
    受信したQR文字列が、どのヒントカード/位置補助情報かを判定する。
    判定優先順位：位置補助情報(2文字) → ヒントカード1(数字+カンマのみ) → ヒントカード2(Base64)
    """
    text = qr_text.strip()

    if len(text) == 2 and text[0] in "ABCD" and text[1] in "1234":
        return HintCardKind.POSITION_ASSIST

    if "," in text and all(
        part.strip().isdigit() for part in text.replace("/", ",").split(",")
    ):
        return HintCardKind.HINT_CARD_1

    if _looks_like_base64(text):
        return HintCardKind.HINT_CARD_2

    return HintCardKind.UNKNOWN


def _looks_like_base64(text: str) -> bool:
    if len(text) == 0 or len(text) % 4 != 0:
        return False
    base64_chars = set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
    )
    return all(c in base64_chars for c in text)


# ==========================================
# 2. QRサーバーへの接続・受信機能
# ==========================================
class QrReaderClient:
    """
    qr_reader.py（QRコード認識サーバー）に接続し、
    検出されたQR文字列を受け取るクライアント。
    """

    def __init__(self, socket_path: str = QR_SOCKET_PATH) -> None:
        self._socket_path = socket_path
        self._sock: socket.socket | None = None

    def connect(self, retry_count: int = 30, retry_interval_sec: float = 0.1) -> None:
        import time

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        connected = False
        for retry in range(retry_count):
            try:
                sock.connect(self._socket_path)
                connected = True
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if retry < retry_count - 1:
                    time.sleep(retry_interval_sec)

        if not connected:
            sock.close()
            raise QrConnectionError(
                f"QRサーバーへの接続に失敗しました: {self._socket_path}"
            )

        sock.send(HANDSHAKE_MSG.encode("utf-8"))
        response = sock.recv(BUFFER_SIZE).decode("utf-8")
        if response != HANDSHAKE_ACK:
            sock.close()
            raise QrConnectionError(f"ハンドシェイクに失敗しました（応答: {response!r}）")

        self._sock = sock
        print("[QrReaderClient] QRサーバーとの接続・ハンドシェイクに成功しました")

    def receive_qr_text(self) -> str | None:
        """
        qr_reader.py の send() に対応する受信処理。
        サーバー側がbufferに書き込んだQR文字列を1件受け取る（ブロッキング）。
        """
        if self._sock is None:
            raise QrConnectionError("connect() を先に呼び出してください")

        try:
            data = self._sock.recv(BUFFER_SIZE)
        except OSError as e:
            raise QrConnectionError(f"受信エラーが発生しました: {e}") from e

        if not data:
            return None

        return data.decode("utf-8")

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None


# ==========================================
# 3. 統合：ゲート位置情報マネージャ
# ==========================================
@dataclass
class GateInfoCollection:
    red_gate: GatePosition | None = None
    other_gates: list[GatePosition] = field(default_factory=list)

    def is_complete(self) -> bool:
        return self.red_gate is not None and len(self.other_gates) >= 2


class GateInfoManager:
    """
    復号キーの保持・QR受信・復号・パースまでを統合管理するクラス。

    使い方の想定：
      1. キャリブレーション開始時に set_decryption_key() で鍵を入力
      2. QRサーバーに connect_qr_server()
      3. ヒントカード1・2をカメラで読み取らせ、その都度 feed_qr_text() に渡す
         （collect_hint_cards() でまとめて待ち受けることも可能）
      4. すべて揃ったら get_gate_info() で結果を取得
      5. ターン終了後 reset() で次ターンに備える
    """

    def __init__(self, socket_path: str = QR_SOCKET_PATH) -> None:
        self._key_store = DecryptionKeyStore()
        self._qr_client = QrReaderClient(socket_path=socket_path)
        self._gate_info = GateInfoCollection()

    # ---- 1. 復号キーの入力 ----
    def set_decryption_key(self, decryption_key: str) -> None:
        self._key_store.set_key(decryption_key)

    # ---- QRサーバーとの接続 ----
    def connect_qr_server(self) -> None:
        self._qr_client.connect()

    # ---- 3. QR文字列の受信と振り分け ----
    def feed_qr_text(self, qr_text: str) -> HintCardKind:
        """
        受信したQR文字列を種類判定し、内部状態に反映する。
        ヒントカード2の場合は、2. で保持しておいた復号キーをここで使用する。
        """
        kind = classify_qr_text(qr_text)

        if kind == HintCardKind.HINT_CARD_1:
            self._gate_info.red_gate = parse_hint_card_1(qr_text)
            print(f"[GateInfoManager] ヒントカード1を取得: {self._gate_info.red_gate}")

        elif kind == HintCardKind.HINT_CARD_2:
            decryption_key = self._key_store.get_key()
            plain_text = decrypt_gate_info(qr_text, decryption_key)
            gates = parse_gate_positions(plain_text)
            self._gate_info.other_gates = gates
            print(f"[GateInfoManager] ヒントカード2を復号: {gates}")

        elif kind == HintCardKind.POSITION_ASSIST:
            print(f"[GateInfoManager] 位置補助情報を検出（本クラスでは未使用）: {qr_text}")

        else:
            print(f"[GateInfoManager] 不明なQR文字列を受信: {qr_text!r}")

        return kind

    # ---- QR受信ループ（ヒントカード2件が揃うまで） ----
    def collect_hint_cards(self, timeout_count: int = 50) -> GateInfoCollection:
        attempts = 0
        while not self._gate_info.is_complete() and attempts < timeout_count:
            qr_text = self._qr_client.receive_qr_text()
            if qr_text is None:
                raise QrConnectionError("QRサーバーとの接続が切断されました")
            self.feed_qr_text(qr_text)
            attempts += 1
        return self._gate_info

    def get_gate_info(self) -> GateInfoCollection:
        return self._gate_info

    def reset(self) -> None:
        """次ターンに備え、復号キーとゲート情報をクリアする。"""
        self._key_store.clear()
        self._gate_info = GateInfoCollection()
        print("[GateInfoManager] 状態をリセットしました（次ターンに備える）")

    def close(self) -> None:
        self._qr_client.close()


if __name__ == "__main__":
    print("=== GateInfoManager 結合テスト（QRサーバーなしのモック版） ===\n")

    def _encrypt_for_test(plain: str, key_str: str) -> str:
        key = _normalize_key(key_str)
        data = plain.encode("utf-8")
        pad_len = AES.block_size - (len(data) % AES.block_size)
        if pad_len == 0:
            pad_len = AES.block_size
        padded = data + bytes([pad_len]) * pad_len
        cipher = AES.new(key, AES.MODE_ECB)
        encrypted = cipher.encrypt(padded)
        return base64.b64encode(encrypted).decode("utf-8")

    test_key = "1234"
    hint1_text = "25,35"
    hint2_plain = "53,54/12,22"
    hint2_cipher = _encrypt_for_test(hint2_plain, test_key)

    print(f"復号キー         : {test_key}")
    print(f"ヒントカード1     : {hint1_text}")
    print(f"ヒントカード2(暗号文): {hint2_cipher}\n")

    manager = GateInfoManager()

    print("--- 1. 復号キーの入力 ---")
    manager.set_decryption_key(test_key)

    print("\n--- 2. ヒントカード1の受信を模擬 ---")
    kind1 = manager.feed_qr_text(hint1_text)
    print(f"判定結果: {kind1}")

    print("\n--- 3. 位置補助情報の受信を模擬 ---")
    kind_assist = manager.feed_qr_text("B3")
    print(f"判定結果: {kind_assist}")

    print("\n--- 4. ヒントカード2（暗号文）の受信を模擬 ---")
    print("    （保持していた復号キーがここで使われる）")
    kind2 = manager.feed_qr_text(hint2_cipher)
    print(f"判定結果: {kind2}")

    print("\n--- 5. 結果確認 ---")
    info = manager.get_gate_info()
    print(f"赤ゲート   : {info.red_gate}")
    print(f"他のゲート : {info.other_gates}")
    print(f"情報が揃ったか: {info.is_complete()}")

    assert info.red_gate is not None
    assert len(info.other_gates) == 2
    print("\n→ ゲート位置情報の取得に成功")

    print("\n--- 6. 次ターンに向けたリセット ---")
    manager.reset()
    info_after_reset = manager.get_gate_info()
    print(f"リセット後 赤ゲート: {info_after_reset.red_gate}")
    print(f"リセット後 情報完了: {info_after_reset.is_complete()}")

    print("\n--- 7. 異常系：キー未設定でヒントカード2を受信 ---")
    try:
        manager.feed_qr_text(hint2_cipher)
        print("NG: エラーが発生すべきところで発生しませんでした")
    except DecryptionKeyError as e:
        print(f"OK: 未設定エラーを検出 -> {e}")

    print("\n全テスト完了")

    # ==========================================
    # 実機での使用イメージ（コメントアウト）
    # ==========================================
    #
    # manager = GateInfoManager()
    # manager.set_decryption_key(received_key_from_operator_panel)
    # manager.connect_qr_server()
    # try:
    #     gate_info = manager.collect_hint_cards(timeout_count=50)
    # except (QrConnectionError, GateDecodeError, DecryptionKeyError) as e:
    #     print(f"ゲート情報の取得に失敗: {e}")
    # else:
    #     if gate_info.is_complete():
    #         pass  # gate_info.red_gate / other_gates を走行制御へ渡す
    # manager.reset()
    # manager.close()
