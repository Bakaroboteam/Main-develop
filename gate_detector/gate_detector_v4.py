#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gate_detector_v4.py — ETロボコン2026 ゲート検出 (Raspberry Pi 4 / USBカメラ / imgsz=320)

v3からの主な変更点
  1. カメラ取得を専用スレッドに分離し、常に「最新フレームだけ」を保持する。
     v3は推論スレッド内で camera.read() していたため、推論に120ms かかる間に
     V4L2ドライバ側のキューにフレームが溜まり、fpsは出ているのに映像が数百ms
     遅れる状態になりやすかった。制御にとってはこの遅延が致命的。
  2. NCNN矩形入力 320x256。640x480を320x320にletterboxすると上下の灰色帯に
     約20%の計算を捨てることになる。高さを256(32の倍数)にして無駄を削る。
  3. 自動露出・自動ホワイトバランスを固定。色相がフレームごとに揺れる現象を止める。
     色誤判定の最大要因はモデルではなくここ。
  4. 色分類を「bbox全体のHSV範囲内ピクセル数」から
     「上部バーROIのHue中央値 → 赤/青/黄への最近傍」に変更。
     - bbox全体だと床と灰色の脚が大半を占め、色が薄まる
     - 範囲内ピクセル数の比較は、色ごとに範囲の広さが違うと不公平になる
     - Hueの円環距離で扱うので、赤の 0/180 またぎを特別扱いしなくてよい
  5. IoUベースの簡易トラッキング + ヒステリシス(N連続ヒットで確定 / M連続ミスで消失)
     と bbox のEMA平滑化。単フレーム判定を直接ステートマシンへ流すのをやめる。
  6. 配信スレッドは描画のみ。競技走行時は STREAM_ENABLED=False でCPUを推論に回す。
  7. /capture で生フレームを保存。学習データ追加用(クリップ単位で分けやすいよう
     セッションIDごとのフォルダに保存する)。

事前準備(重要):
    yolo export model=runs/detect/train/weights/best.pt format=ncnn imgsz=[256,320]
  320x320でエクスポートしたモデルのままでは矩形入力の高速化は効きません。

起動:
    python3 gate_detector_v4.py
    ブラウザで http://<PiのIP>:8000/
"""

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
# NCNNが全コアを使えるよう、OpenCV側のスレッドは絞ってコア競合を避ける
os.environ.setdefault("OMP_NUM_THREADS", "4")

import json
import subprocess
import threading
import time
from collections import Counter, deque
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from ultralytics import YOLO

cv2.setNumThreads(2)

# =============================================================================
# 設定
# =============================================================================

MODEL_PATH = "/home/pi/gate-yolo/best_ncnn_model"

# --- カメラ ---
DEVICE = "/dev/video1"
CAM_W, CAM_H = 640, 480
CAM_FPS = 15          # 推論が8〜10fpsなので15で十分。30にするとMJPGデコードでCPUを食う
EXPOSURE_ABS = 200    # コース照明下で床が白飛びしない範囲の最大値に合わせ込む
WB_TEMP = 4600        # 会場照明の色温度。校正時に一度決めて固定する

# --- 推論 ---
IMGSZ = [320, 320]    # [height, width]。エクスポート時と必ず一致させる
CONF_TH = 0.60        # F1ピーク 0.604（train-5採用時の0.55から変更）
IOU_TH = 0.45         # NMS。v3の既定0.7は緩く、同一ゲートに重複枠が出ていた
MAX_DET = 5
MIN_BOX_AREA = 400    # これ未満の極小boxはノイズとして捨てる(px^2)

# --- 色分類 ---
# コの字ゲートは上部の横棒と支柱上側に色ラベルが巻かれ、下側は灰色の脚。
# よってbboxの上部だけを見る。
ROI_TOP_FRAC = 0.30   # bbox高さの上から30%
ROI_SIDE_FRAC = 0.20  # 左右それぞれ20%を除外(=中央60%を使う)
S_MIN = 60            # 彩度がこれ未満の画素は「色なし」として除外(床・灰色の脚)
V_MIN = 40            # 暗すぎる画素を除外
V_MAX = 245           # 白飛び画素を除外
MIN_VALID_RATIO = 0.08  # ROI面積に対する有効画素の下限比
MIN_VALID_PX = 30       # 有効画素の絶対下限
MAX_HUE_DIST = 22       # どの基準色からもこれ以上離れていたら色不明

# 基準Hue(OpenCVのHは0〜179)。/calibrate で実測値に上書きされる
DEFAULT_HUE_REFS = {"red": 0, "yellow": 27, "blue": 108}
REFS_FILE = "hue_refs.json"

# --- トラッキング / ヒステリシス ---
TRACK_IOU_TH = 0.30
CONFIRM_HITS = 3      # 3フレーム連続で確定。10fpsなら0.3秒
MAX_MISSES = 5        # 5フレーム連続で見失ったら破棄
COLOR_VOTE_N = 7      # 直近7フレームの色を多数決
BBOX_EMA = 0.4        # 新しい観測の重み。小さいほど滑らかだが追従が遅れる

# --- 配信(デバッグ用) ---
STREAM_ENABLED = True   # 競技走行時は False にしてCPUを推論へ回す
STREAM_FPS = 5.0
STREAM_W, STREAM_H = 320, 240

# --- 制御 ---
TARGET_COLOR = None     # /target/<color> で実行中に変更可能

# --- データ収集 ---
DATASET_DIR = "/home/pi/gate-yolo/captures"

DRAW_COLOR = {"red": (0, 0, 255), "blue": (255, 0, 0), "yellow": (0, 255, 255)}

# 起動ごとに1フォルダ。あとで train/val をクリップ単位で分割できるようにするため
SESSION_DIR = os.path.join(DATASET_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))


# =============================================================================
# カメラ: V4L2制御 + 最新フレーム保持スレッド
# =============================================================================

def _v4l2_set(device, pairs):
    """v4l2-ctlでカメラ制御を叩く。カーネル/カメラによって項目名が異なるため、
       新旧両方の名前を試して、通ったものだけ効けばよいという方針にする。"""
    applied = []
    for name, value in pairs:
        try:
            r = subprocess.run(
                ["v4l2-ctl", "-d", device, "-c", f"{name}={value}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3,
            )
            if r.returncode == 0:
                applied.append(name)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return applied
    return applied


def configure_camera(device):
    """自動露出・自動WBを切って固定する。
       これを入れないと、走行体が動くたびにカメラが露出とWBを再計算し、
       同じゲートのHueがフレームごとに数十度ずれる。色判定が不安定な原因の大半。"""
    # auto_exposure: 1=manual, 3=aperture priority(新名称)
    # exposure_auto : 1=manual, 3=aperture priority(旧名称)
    applied = _v4l2_set(device, [
        ("auto_exposure", 1),
        ("exposure_time_absolute", EXPOSURE_ABS),
        ("white_balance_automatic", 0),
        ("white_balance_temperature", WB_TEMP),
    ])
    applied += _v4l2_set(device, [
        ("exposure_auto", 1),
        ("exposure_absolute", EXPOSURE_ABS),
        ("white_balance_temperature_auto", 0),
    ])
    return applied


class CameraReader(threading.Thread):
    """カメラから読み続け、常に最新の1枚だけを保持するスレッド。

    このスレッドが止まらない限りドライバ側のキューは進み続けるので、
    推論がどれだけ遅くても『取得した瞬間の映像』が読める。
    v3のように推論ループ内で read() すると、推論時間ぶんキューが伸びて遅延が育つ。
    """

    def __init__(self, device):
        super().__init__(daemon=True)
        self.device = device
        self.cap = None
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.cam_fps = 0.0
        self.open()

    def open(self):
        if self.cap is not None:
            self.cap.release()
        configure_camera(self.device)
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.cap = cap
        return cap.isOpened()

    def run(self):
        prev = time.time()
        fps = 0.0
        fail = 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                fail += 1
                time.sleep(0.05)
                if fail >= 20:       # USBの抜けや一時的なエラーから復帰させる
                    self.open()
                    fail = 0
                continue
            fail = 0

            now = time.time()
            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            with self._lock:
                self._frame = frame
                self._seq += 1
                self.cam_fps = fps

    def read(self):
        """(seq, frame) を返す。frameは呼び出し側が書き換えない前提の参照。
           描画する場合は必ず copy() すること。"""
        with self._lock:
            return self._seq, self._frame

    def stop(self):
        self._stop.set()


# =============================================================================
# 色分類
# =============================================================================

hue_refs = dict(DEFAULT_HUE_REFS)
calib_info = {}          # color -> {"hue","s","v","at"} 校正時の実測値(表示用)
refs_lock = threading.Lock()


def load_refs():
    global hue_refs
    if not os.path.exists(REFS_FILE):
        return
    try:
        with open(REFS_FILE) as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return
    with refs_lock:
        for color, v in raw.items():
            if color in DRAW_COLOR and isinstance(v, dict) and "hue" in v:
                hue_refs[color] = int(v["hue"])
                calib_info[color] = v


def save_refs():
    with refs_lock:
        raw = {c: calib_info.get(c, {"hue": hue_refs[c]}) for c in hue_refs}
    with open(REFS_FILE, "w") as f:
        json.dump(raw, f, ensure_ascii=False, indent=1)


def hue_distance(a, b):
    """Hueの円環距離。OpenCVのHは0〜179なので、周期180で折り返す。
       これにより赤(0付近と179付近に分裂する)を2レンジに分けて扱う必要がなくなる。"""
    d = abs(float(a) - float(b)) % 180.0
    return min(d, 180.0 - d)


def _hue_median(frame, x1, y1, x2, y2):
    """指定矩形の有効画素からHueの中央値と有効画素数を返す。
       平均でなく中央値を使うのは、背景が少し混入しても中心値がずれないため。"""
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None, 0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.int16)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = (s >= S_MIN) & (v >= V_MIN) & (v <= V_MAX)
    n = int(np.count_nonzero(mask))
    if n == 0:
        return None, 0
    return float(np.median(h[mask])), n


def classify_color(frame, box):
    """bbox上部のバー部分から色を判定する。

    戻り値: (color or None, hue or None, valid_px)
    判定は「3色への最近傍」。しきい値表を持たないので、照明が多少変わっても
    3色の相対関係が保たれる限り正しく分類できる。
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    side = int(w * ROI_SIDE_FRAC)
    rx1, rx2 = x1 + side, x2 - side
    ry1, ry2 = y1, y1 + max(int(h * ROI_TOP_FRAC), 4)
    if rx2 - rx1 < 3:
        rx1, rx2 = x1, x2

    hue, n = _hue_median(frame, rx1, ry1, rx2, ry2)
    need = max(MIN_VALID_PX, int((rx2 - rx1) * (ry2 - ry1) * MIN_VALID_RATIO))

    # 上部ROIで色が拾えないケース(ゲートが傾いている/上端が切れている)は
    # bbox全体にフォールバックする。精度は落ちるが色不明を連発するよりまし。
    if hue is None or n < need:
        hue, n = _hue_median(frame, x1, y1, x2, y2)
        need = max(MIN_VALID_PX, int(w * h * MIN_VALID_RATIO))
        if hue is None or n < need:
            return None, hue, n

    with refs_lock:
        refs = dict(hue_refs)
    best = min(refs, key=lambda c: hue_distance(hue, refs[c]))
    if hue_distance(hue, refs[best]) > MAX_HUE_DIST:
        return None, hue, n
    return best, hue, n


# =============================================================================
# トラッキング + ヒステリシス
# =============================================================================

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


class Track:
    """1つのゲートに対する追跡状態。

    - bbox は EMA で平滑化する(フレーム間の枠のガタつきを抑える)
    - 色は直近COLOR_VOTE_Nフレームの多数決で決める(単フレームの誤判定を吸収)
    - CONFIRM_HITS 連続でヒットするまで confirmed にしない(FPが制御に漏れない)
    - MAX_MISSES 連続で見失うまで消さない(1〜2フレームの欠落で状態が飛ばない)
    """

    _next_id = 1

    def __init__(self, bbox, conf, color):
        self.id = Track._next_id
        Track._next_id += 1
        self.bbox = list(map(float, bbox))
        self.conf = conf
        self.hits = 1
        self.misses = 0
        self.color_hist = deque([color], maxlen=COLOR_VOTE_N)

    def update(self, bbox, conf, color):
        for i in range(4):
            self.bbox[i] = (1 - BBOX_EMA) * self.bbox[i] + BBOX_EMA * float(bbox[i])
        self.conf = 0.6 * self.conf + 0.4 * conf
        self.hits += 1
        self.misses = 0
        self.color_hist.append(color)

    def mark_missed(self):
        self.misses += 1

    @property
    def alive(self):
        return self.misses <= MAX_MISSES

    @property
    def confirmed(self):
        return self.hits >= CONFIRM_HITS and self.misses == 0

    @property
    def color(self):
        votes = [c for c in self.color_hist if c is not None]
        if not votes:
            return None
        color, cnt = Counter(votes).most_common(1)[0]
        # 過半数に届かない=色が揺れている → 確定させない
        return color if cnt * 2 > len(self.color_hist) else None

    def as_dict(self):
        x1, y1, x2, y2 = (int(round(v)) for v in self.bbox)
        return {
            "id": self.id,
            "color": self.color,
            "bbox": [x1, y1, x2, y2],
            "cx": (x1 + x2) // 2,
            "cy": (y1 + y2) // 2,
            "area": (x2 - x1) * (y2 - y1),
            "err_px": (x1 + x2) // 2 - CAM_W // 2,
            "conf": round(float(self.conf), 3),
            "confirmed": self.confirmed,
            "hits": self.hits,
            "misses": self.misses,
        }


class Tracker:
    def __init__(self):
        self.tracks = []

    def update(self, detections):
        """IoUの大きい順に貪欲マッチング。ゲートは同時に数個しか映らないので
           ハンガリアン法まで持ち出す必要はなく、貪欲で十分かつ軽い。"""
        pairs = []
        for ti, t in enumerate(self.tracks):
            for di, d in enumerate(detections):
                v = iou(t.bbox, d["bbox"])
                if v >= TRACK_IOU_TH:
                    pairs.append((v, ti, di))
        pairs.sort(reverse=True)

        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            d = detections[di]
            self.tracks[ti].update(d["bbox"], d["conf"], d["color"])
            used_t.add(ti)
            used_d.add(di)

        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.mark_missed()
        for di, d in enumerate(detections):
            if di not in used_d:
                self.tracks.append(Track(d["bbox"], d["conf"], d["color"]))

        self.tracks = [t for t in self.tracks if t.alive]
        return self.tracks


def pick_target(tracks, target_color):
    """制御用に1つへ潰す: 確定済み・目標色・最大area(=最も近い)。"""
    if not target_color:
        return None
    cands = [t for t in tracks if t.confirmed and t.color == target_color]
    if not cands:
        return None
    return max(cands, key=lambda t: t.as_dict()["area"])


# =============================================================================
# 推論
# =============================================================================

model = YOLO(MODEL_PATH, task="detect")
_dummy = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
model.predict(_dummy, imgsz=IMGSZ, verbose=False)   # ウォームアップ


def detect(frame):
    res = model.predict(frame, imgsz=IMGSZ, conf=CONF_TH, iou=IOU_TH,
                        max_det=MAX_DET, verbose=False)
    boxes = res[0].boxes
    out = []
    if boxes is None or len(boxes) == 0:
        return out
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    for (x1, y1, x2, y2), cf in zip(xyxy, confs):
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(CAM_W, int(x2)); y2 = min(CAM_H, int(y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
            continue
        color, hue, npx = classify_color(frame, (x1, y1, x2, y2))
        out.append({"bbox": (x1, y1, x2, y2), "conf": float(cf),
                    "color": color, "hue": hue, "valid_px": npx})
    return out


# ---- 共有状態 ----
state_lock = threading.Lock()
latest_frame = None
latest_tracks = []       # as_dict() 済みのリスト
latest_target = None
latest_error = None
det_fps = 0.0
infer_ms = 0.0

cam = CameraReader(DEVICE)
tracker = Tracker()


def inference_worker():
    """カメラの最新フレームを取り、推論→色分類→追跡までを回すスレッド。
       制御(SPIKE連携)は get_target() / latest_tracks を読むだけでよい。"""
    global latest_frame, latest_tracks, latest_target, latest_error, det_fps, infer_ms
    prev_seq = -1
    prev_t = time.time()
    fps = 0.0
    ms = 0.0

    while True:
        seq, frame = cam.read()
        if frame is None or seq == prev_seq:
            time.sleep(0.005)      # 新しいフレームが来るまで待つ(同じ絵を2度推論しない)
            continue
        prev_seq = seq

        t0 = time.time()
        try:
            dets = detect(frame)
            err = None
        except Exception as e:      # 推論が落ちても走行を止めないよう握り潰して記録
            dets = []
            err = f"{type(e).__name__}: {e}"

        tracks = tracker.update(dets)
        target = pick_target(tracks, TARGET_COLOR)

        now = time.time()
        ms = 0.9 * ms + 0.1 * (now - t0) * 1000.0
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        with state_lock:
            latest_frame = frame
            latest_tracks = [t.as_dict() for t in tracks]
            latest_target = target.as_dict() if target else None
            latest_error = err
            det_fps = fps
            infer_ms = ms


def get_target():
    """制御側から呼ぶ想定のAPI。確定済みの目標ゲート(dict)かNoneを返す。
       err_px が画面中心からの左右ずれ(px)なので、そのまま操舵に使える。"""
    with state_lock:
        return None if latest_target is None else dict(latest_target)


# =============================================================================
# 配信 / Web
# =============================================================================

app = Flask(__name__)


def draw_overlay(frame, tracks, target_id, err, fps, ms, cfps):
    cv2.line(frame, (CAM_W // 2, 0), (CAM_W // 2, CAM_H), (128, 128, 128), 1)

    for r in tracks:
        x1, y1, x2, y2 = r["bbox"]
        c = DRAW_COLOR.get(r["color"], (200, 200, 200))
        is_target = target_id is not None and r["id"] == target_id
        thickness = 4 if is_target else (2 if r["confirmed"] else 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, thickness)
        cv2.circle(frame, (r["cx"], r["cy"]), 5, c, -1)

        # 色をサンプリングしているROIを可視化(校正のあたりを付けやすくする)
        side = int((x2 - x1) * ROI_SIDE_FRAC)
        cv2.rectangle(frame, (x1 + side, y1),
                      (x2 - side, y1 + max(int((y2 - y1) * ROI_TOP_FRAC), 4)),
                      (255, 255, 255), 1)

        ty = y1 - 8 if y1 > 34 else y2 + 16
        head = "TARGET " if is_target else ("" if r["confirmed"] else "? ")
        cv2.putText(frame, f"#{r['id']} {head}{r['color']} {r['conf']:.2f} err={r['err_px']}",
                    (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        cv2.putText(frame, f"area={r['area']} hit={r['hits']} miss={r['misses']}",
                    (x1, ty + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

    cv2.putText(frame, f"det {fps:4.1f}fps / infer {ms:5.1f}ms / cam {cfps:4.1f}fps",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"target={TARGET_COLOR}", (8, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if err:
        cv2.putText(frame, f"DETECT ERROR: {err[:48]}", (8, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    missing = [c for c in DRAW_COLOR if c not in calib_info]
    if missing:
        cv2.putText(frame, f"NOT CALIBRATED: {','.join(missing)}",
                    (8, CAM_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    return frame


def generate_frames():
    interval = 1.0 / STREAM_FPS
    while True:
        t0 = time.time()
        if not STREAM_ENABLED:
            time.sleep(0.5)
            continue

        with state_lock:
            frame = None if latest_frame is None else latest_frame.copy()
            tracks = list(latest_tracks)
            target = latest_target
            err, fps, ms = latest_error, det_fps, infer_ms
        if frame is None:
            time.sleep(0.1)
            continue

        frame = draw_overlay(frame, tracks, target["id"] if target else None,
                             err, fps, ms, cam.cam_fps)
        small = cv2.resize(frame, (STREAM_W, STREAM_H))
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")

        elapsed = time.time() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)


INDEX_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>gate detector</title>
<style>body{font-family:sans-serif;margin:12px}
button{padding:6px 12px;margin:2px;font-size:14px}
#st{margin-left:8px}</style></head><body>
<div>
  <b>色校正:</b>
  <button onclick="sel('red')">赤</button>
  <button onclick="sel('blue')">青</button>
  <button onclick="sel('yellow')">黄</button>
  <span id="st">色ボタンを押してから、映像内のゲートの<b>色が付いた部分</b>をクリック</span>
</div>
<div>
  <b>目標色:</b>
  <button onclick="tgt('red')">赤</button>
  <button onclick="tgt('blue')">青</button>
  <button onclick="tgt('yellow')">黄</button>
  <button onclick="tgt('none')">解除</button>
  <button onclick="cap()">フレーム保存</button>
</div>
<img id="v" src="/video" style="cursor:crosshair;width:640px;height:480px">
<script>
let color=null;
const st=document.getElementById('st');
function sel(c){color=c;st.textContent=c+' を校正中: ゲートの色部分をクリック';}
async function tgt(c){const j=await(await fetch('/target/'+c,{method:'POST'})).json();st.textContent=j.msg;}
async function cap(){const j=await(await fetch('/capture',{method:'POST'})).json();st.textContent=j.msg;}
document.getElementById('v').onclick=async(e)=>{
  if(!color)return;
  const r=e.target.getBoundingClientRect();
  const x=Math.round((e.clientX-r.left)*640/r.width);
  const y=Math.round((e.clientY-r.top)*480/r.height);
  const j=await(await fetch(`/calibrate/${color}?x=${x}&y=${y}`,{method:'POST'})).json();
  st.textContent=j.msg; color=null;
};
</script></body></html>"""


@app.route("/")
def index():
    return INDEX_HTML


@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/calibrate/<color>", methods=["POST"])
def calibrate(color):
    """クリック位置周辺のHue中央値を、その色の基準Hueとして記録する。
       v3のように上下限レンジを作らないので、赤の0/180またぎの分岐が不要になった。"""
    if color not in DRAW_COLOR:
        return jsonify(msg="不明な色"), 400
    try:
        x, y = int(request.args["x"]), int(request.args["y"])
    except (KeyError, ValueError):
        return jsonify(msg="座標が不正"), 400

    with state_lock:
        frame = None if latest_frame is None else latest_frame.copy()
    if frame is None:
        return jsonify(msg="フレーム未取得"), 500

    half = 12
    x0, x1 = max(x - half, 0), min(x + half, CAM_W)
    y0, y1 = max(y - half, 0), min(y + half, CAM_H)
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    s_med = int(np.median(hsv[:, :, 1]))
    v_med = int(np.median(hsv[:, :, 2]))
    if s_med < S_MIN:
        return jsonify(msg=f"彩度が低すぎます(S={s_med})。色の付いた部分をクリックしてください"), 400

    # 彩度の高い画素だけでHueを取る(縁の灰色が混ざると中央値がずれる)
    mask = (hsv[:, :, 1] >= S_MIN) & (hsv[:, :, 2] >= V_MIN) & (hsv[:, :, 2] <= V_MAX)
    h_med = int(np.median(hsv[:, :, 0][mask]))

    with refs_lock:
        hue_refs[color] = h_med
        calib_info[color] = {"hue": h_med, "s": s_med, "v": v_med,
                             "at": datetime.now().isoformat(timespec="seconds")}
    save_refs()

    with refs_lock:
        others = {c: h for c, h in hue_refs.items() if c != color}
    nearest = min(others, key=lambda c: hue_distance(h_med, others[c])) if others else None
    warn = ""
    if nearest and hue_distance(h_med, others[nearest]) < MAX_HUE_DIST * 1.5:
        d = hue_distance(h_med, others[nearest])
        warn = f" ※{nearest}とHueが近い({d:.0f}度)。露出/WB設定を見直してください"
    return jsonify(msg=f"{color} 校正完了 H={h_med} S={s_med} V={v_med}{warn}")


@app.route("/target/<color>", methods=["POST", "GET"])
def set_target(color):
    global TARGET_COLOR
    if color in ("none", "null", ""):
        TARGET_COLOR = None
        return jsonify(msg="目標色を解除しました", target=None)
    if color not in DRAW_COLOR:
        return jsonify(msg="不明な色"), 400
    TARGET_COLOR = color
    return jsonify(msg=f"目標色を {color} に設定", target=color)


@app.route("/capture", methods=["POST"])
def capture():
    """生フレームを保存する。学習データ追加用。
       起動ごとのセッションフォルダに入れるので、あとで train/val を
       『クリップ単位』で分割しやすい(フレーム単位分割はリークの原因になる)。"""
    with state_lock:
        frame = None if latest_frame is None else latest_frame.copy()
    if frame is None:
        return jsonify(msg="フレーム未取得"), 500
    os.makedirs(SESSION_DIR, exist_ok=True)
    name = f"gate_{datetime.now().strftime('%Y%m%d%H%M%S_%f')}.jpg"
    path = os.path.join(SESSION_DIR, name)
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return jsonify(msg=f"保存: {path}")


@app.route("/status")
def status():
    with state_lock:
        return jsonify(
            det_fps=round(det_fps, 1),
            infer_ms=round(infer_ms, 1),
            cam_fps=round(cam.cam_fps, 1),
            target_color=TARGET_COLOR,
            target=latest_target,
            error=latest_error,
            hue_refs=hue_refs,
            tracks=latest_tracks,
        )


if __name__ == "__main__":
    load_refs()
    cam.start()
    threading.Thread(target=inference_worker, daemon=True).start()
    app.run(host="0.0.0.0", port=8000, threaded=True)
