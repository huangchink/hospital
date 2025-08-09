# -*- coding: utf-8 -*-
import logging
import gazefollower
gazefollower.logging = logging
import gazefollower.face_alignment.MediaPipeFaceAlignment as mpa
mpa.logging = logging
import os
import cv2
import pygame
import numpy as np
import time
import sys
import math
import random
from collections import deque
import tkinter as tk
from gazefollower import GazeFollower
from gazefollower.misc import DefaultConfig
from tkinter import ttk, colorchooser, filedialog
from pathlib import Path
from gazefollower.calibration import SVRCalibration
from gazefollower.logger import Log as GFLog



def _profile_dir(cfg, W, H):
    # 存在專案根目錄(這支 .py 所在資料夾)底下的 calibration_profiles
    project_root = Path(__file__).resolve().parent
    base = project_root / "calibration_profiles"
    base.mkdir(parents=True, exist_ok=True)

    name = (cfg.get('user_name') or "default").strip().replace(" ", "_")
    return base / f"{name}_{cfg['calib_pts']}pt_{W}x{H}"

DO_BLUR    = True        # True: 輕微高斯模糊，減少鋸齒
BLUR_KSIZE = (5, 5)
BLUR_SIGMA = 1.0

# ---------- Utils ----------
def restore_event_filter():
    # 允許所有事件（None = 關閉過濾）
    try:
        pygame.event.set_allowed(None)
        pygame.event.clear()
        pygame.event.pump()
    except Exception:
        pass

def prep_input_for_calibration():
    # 1) 關掉文字輸入/IME，避免空白被當成選字
    try:
        pygame.key.stop_text_input()
    except Exception:
        pass
    # 2) 清掉卡住的修飾鍵狀態（Shift/Ctrl/Alt）
    try:
        pygame.key.set_mods(0)
    except Exception:
        pass
    # 3) 事件佇列清空，只保留關鍵事件
    pygame.event.set_allowed(None)
    pygame.event.set_allowed([pygame.KEYDOWN, pygame.KEYUP, pygame.QUIT,
                              pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP,
                              pygame.ACTIVEEVENT])
    pygame.event.clear()
    pygame.event.pump()
    # 4) 把鍵盤/滑鼠焦點鎖到這個視窗（可選）
    try:
        pygame.event.set_grab(True)
    except Exception:
        pass

def ensure_pygame_focus(timeout=2.0):
    t0 = time.time()
    while not pygame.key.get_focused():
        pygame.event.pump()
        if time.time() - t0 > timeout:
            break
        time.sleep(0.02)

def to_rgb_tuple(rgb_like):
    return tuple(int(v) for v in rgb_like)

def screen_width_deg_from_cm(width_cm: float, dist_cm: float) -> float:
    """由螢幕可視寬度與觀看距離換算『螢幕橫向視角(度)』"""
    if dist_cm <= 0 or width_cm <= 0:
        return 0.0
    return 2.0 * math.degrees(math.atan((width_cm / 2.0) / dist_cm))

# ---------- VA 分數對應（以 cpd 判斷） ----------
# def get_va_result_cpd(cpd_value: float, screen_width_deg: float):
#     if screen_width_deg <= 0:
#         return "Unknown"
#     thr_cs = [900,800,700,600,500,400,300,200,100]
#     scores = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
#     thr_cpd = [t / screen_width_deg for t in thr_cs]
#     for t, s in zip(thr_cpd, scores):
#         if cpd_value >= t:
#             return s
#     return "Unknown"
def get_va_result_cpd(cpd_value: float, *_ignored):
    """
    只依據 cpd 判定 VA 分數。你可依需求微調 thr_cpd。
    """
    # cpd 門檻（約 36,32,28,24,20,16,12,8,4）
    thr_cpd = [18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0]
    scores  = [0.9,  0.8,  0.7,  0.6,  0.5,  0.4,  0.3, 0.2, 0.1]

    for t, s in zip(thr_cpd, scores):
        if cpd_value >= t:
            return s
    return "Unknown"

# ---------- 另一個圓顏色 = 亮/暗顏色分量平均 ----------
def mean_color_rgb(light, dark):
    l = np.array(light, dtype=np.uint16)
    d = np.array(dark,  dtype=np.uint16)
    return tuple(((l + d) // 2).astype(np.uint8))

# ---------- 產生「局部貼圖」用的座標網格與圓形 alpha mask ----------
def prepare_patch_grid(rad):
    diam = 2 * rad
    yy, xx = np.mgrid[0:diam, 0:diam]
    xx = xx - rad
    yy = yy - rad
    circle_mask = (xx * xx + yy * yy) <= (rad * rad)   # boolean
    return xx.astype(np.float32), yy.astype(np.float32), circle_mask

# ---------- 以角度直接生成條紋（只算局部貼圖） ----------
def generate_grating_oriented_patch(freq_cycles_per_screen, xx, yy, angle_deg, w_total_px,
                                    color_dark=(0,0,0),
                                    color_light=(255,255,255),
                                    do_blur=True):
    """
    freq_cycles_per_screen: 『整個螢幕寬』內的周期數（cycles/screen）
    w_total_px: 螢幕寬度(像素)
    """
    theta = np.deg2rad(angle_deg)
    # 在旋轉後的軸上投影：u = x*cosθ + y*sinθ
    u = xx * np.cos(theta) + yy * np.sin(theta)
    # 正弦波：螢幕寬 w_total_px 內有 freq_cycles_per_screen 個周期
    g = 0.5 + 0.5 * np.sin(2 * np.pi * freq_cycles_per_screen * u / float(w_total_px))  # 0~1

    gray = (g * 255).astype(np.uint8)
    if do_blur:
        gray = cv2.GaussianBlur(gray, BLUR_KSIZE, sigmaX=BLUR_SIGMA)

    a = gray.astype(np.float32) / 255.0
    light = np.array(color_light, dtype=np.float32)
    dark  = np.array(color_dark,  dtype=np.float32)
    out = (a[..., None] * light + (1 - a[..., None]) * dark).astype(np.uint8)  # (H,W,3) RGB
    return out

# ---------- Staircase（cpd） ----------
class Staircase:
    def __init__(self, start, step, minv, maxv):
        self.freq = float(start)          # 這裡的 freq 其實就是 cpd
        self.step = float(step)
        self.minv, self.maxv = float(minv), float(maxv)
        self.reversals = []
        self.last_correct = None
        self.correct_streak = 0
        self.max_correct_streak = 0
        self.incorrect_streak = 0         # ← 連續答錯計數

    def update(self, correct):
        if self.last_correct is not None and correct != self.last_correct:
            self.reversals.append(self.freq)
        self.last_correct = correct

        if correct:
            self.correct_streak += 1
            self.max_correct_streak = max(self.max_correct_streak, self.correct_streak)
            self.incorrect_streak = 0     # ← 答對就清空錯誤連擊
        else:
            self.correct_streak = 0
            self.incorrect_streak += 1    # ← 連續答錯+1

        delta = self.step if correct else -self.step
        self.freq = min(self.maxv, max(self.minv, self.freq + delta))

    def done(self):
        return (len(self.reversals) >= 4) or \
               (self.freq >= self.maxv and self.max_correct_streak >= 3) or \
               (self.incorrect_streak >= 4)   # ← 新增條件：連續 4 次錯誤就結束


# ---------- GUI ----------
class SettingsWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VA Test Settings")
        self.resizable(False, False)
        self.geometry("1020x1600")

        self.cali_img_path_var = tk.StringVar(value="")
        self.cali_img_w_var    = tk.IntVar(value=170)
        self.cali_img_h_var    = tk.IntVar(value=170)

        LABEL_FONT = ("Arial", 14)
        ENTRY_FONT = ("Arial", 14)

        # 基本參數
        self.user_var = tk.StringVar(value="anonymous")
        self.gaze_color_var  = tk.StringVar(value="0,255,0")  # 視線點顏色（綠）
        self.gaze_radius_var = tk.IntVar(value=30)            # 視線點半徑(px)
        self.gaze_width_var  = tk.IntVar(value=4)             # 視線點線寬(0=實心)
        self.calib_var  = tk.StringVar(value="5")
        self.stim_var   = tk.DoubleVar(value=5.0)
        self.blank_var  = tk.DoubleVar(value=1.0)
        self.rad_var    = tk.IntVar(value=400)
        self.thresh_var = tk.DoubleVar(value=400)
        self.rotate_var = tk.BooleanVar(value=False)
        self.rot_speed_var = tk.DoubleVar(value=1.0)   # deg/s
        self.rot_dir_var   = tk.StringVar(value="CW")  # CW / CCW

        # 顏色
        self.color_light_var = tk.StringVar(value="255,255,255")
        self.color_dark_var  = tk.StringVar(value="0,0,0")
        self.bg_color_var    = tk.StringVar(value="0,0,0")

        # 螢幕與距離（cm）
        self.scr_width_cm_var = tk.DoubleVar(value=53.0)
        self.view_dist_cm_var = tk.DoubleVar(value=120.0)

        # Staircase（以 cpd）
        self.start_cpd_var = tk.DoubleVar(value=2.0)
        self.step_cpd_var  = tk.DoubleVar(value=2.0)
        self.min_cpd_var   = tk.DoubleVar(value=1.0)
        self.max_cpd_var   = tk.DoubleVar(value=20.0)

        # 可選：外部 checkpoint 目錄
        self.calib_dir_var = tk.StringVar(value="")

        self.cfg = None
        pad = {'padx': 10, 'pady': 8}

        # 版面
        r = 0
        ttk.Label(self, text="User name:", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.user_var, font=ENTRY_FONT, width=15).grid(row=r, column=1, **pad); r += 1

        ttk.Label(self, text="Calibration Points:", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Combobox(self, textvariable=self.calib_var,
                     values=["5", "9", "13"], state="readonly", width=10, font=ENTRY_FONT)\
            .grid(row=r, column=1, **pad); r += 1

        # 外部 checkpoint 目錄（可選）
        ttk.Label(self, text="Calibration folder (optional):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        cdir_frame = ttk.Frame(self); cdir_frame.grid(row=r, column=1, sticky="w", **pad)
        ttk.Entry(cdir_frame, textvariable=self.calib_dir_var, font=ENTRY_FONT, width=28).pack(side="left")
        def _browse_calib_dir():
            p = filedialog.askdirectory(title="Choose calibration folder")
            if p:
                self.calib_dir_var.set(p)
        ttk.Button(self, text="Browse", command=_browse_calib_dir).grid(row=r, column=2, **pad); r += 1

        ttk.Label(self, text="Stimulus Duration (s):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(self, textvariable=self.stim_var, from_=0.5, to=10.0,
                    increment=0.1, width=10, font=ENTRY_FONT)\
            .grid(row=r, column=1, **pad); r += 1

        ttk.Label(self, text="Blank Duration (s):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(self, textvariable=self.blank_var, from_=0.5, to=10.0,
                    increment=0.1, width=10, font=ENTRY_FONT)\
            .grid(row=r, column=1, **pad); r += 1

        ttk.Label(self, text="Circle Radius (px):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(self, textvariable=self.rad_var, from_=50, to=800,
                    increment=10, width=10, font=ENTRY_FONT)\
            .grid(row=r, column=1, **pad); r += 1

        ttk.Label(self, text="Threshold (px):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(self, textvariable=self.thresh_var, from_=10, to=1000,
                    increment=10, width=10, font=ENTRY_FONT)\
            .grid(row=r, column=1, **pad); r += 1

        def choose_color(target_var):
            color = colorchooser.askcolor()[0]
            if color:
                r_, g_, b_ = [int(c) for c in color]
                target_var.set(f"{r_},{g_},{b_}")

        ttk.Label(self, text="Bright Stripe Color:", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.color_light_var, font=ENTRY_FONT, width=15).grid(row=r, column=1, **pad)
        ttk.Button(self, text="Pick", command=lambda: choose_color(self.color_light_var)).grid(row=r, column=2, **pad)
        r += 1

        ttk.Label(self, text="Dark Stripe Color:", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.color_dark_var, font=ENTRY_FONT, width=15).grid(row=r, column=1, **pad)
        ttk.Button(self, text="Pick", command=lambda: choose_color(self.color_dark_var)).grid(row=r, column=2, **pad)
        r += 1

        ttk.Label(self, text="Background Color:", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.bg_color_var, font=ENTRY_FONT, width=15).grid(row=r, column=1, **pad)
        ttk.Button(self, text="Pick", command=lambda: choose_color(self.bg_color_var)).grid(row=r, column=2, **pad)
        r += 1

        # Calibration image
        ttk.Label(self, text="Calibration image:", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        path_frame = ttk.Frame(self); path_frame.grid(row=r, column=1, sticky="w", **pad)
        ttk.Entry(path_frame, textvariable=self.cali_img_path_var, font=ENTRY_FONT, width=28).pack(side="left")
        def _browse_cali_img():
            p = filedialog.askopenfilename(
                title="Choose calibration image",
                filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
            )
            if p:
                self.cali_img_path_var.set(p)
        ttk.Button(self, text="Browse", command=_browse_cali_img).grid(row=r, column=2, **pad); r += 1

        # Calibration image size (px)
        ttk.Label(self, text="Calibration image size (px):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        size_frame = ttk.Frame(self); size_frame.grid(row=r, column=1, sticky="w", **pad)
        ttk.Spinbox(size_frame, textvariable=self.cali_img_w_var, from_=10, to=800,
                    increment=5, width=6, font=ENTRY_FONT).pack(side="left")
        ttk.Label(size_frame, text=" x ", font=LABEL_FONT).pack(side="left")
        ttk.Spinbox(size_frame, textvariable=self.cali_img_h_var, from_=10, to=800,
                    increment=5, width=6, font=ENTRY_FONT).pack(side="left")
        r += 1

        # Gaze marker color / size
        ttk.Label(self, text="Gaze marker color:", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.gaze_color_var, font=ENTRY_FONT, width=15).grid(row=r, column=1, **pad)
        ttk.Button(self, text="Pick", command=lambda: choose_color(self.gaze_color_var)).grid(row=r, column=2, **pad)
        r += 1

        ttk.Label(self, text="Gaze marker radius (px):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(self, textvariable=self.gaze_radius_var, from_=1, to=200,
                    increment=1, width=10, font=ENTRY_FONT).grid(row=r, column=1, **pad); r += 1

        ttk.Label(self, text="Gaze marker line width:", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(self, textvariable=self.gaze_width_var, from_=0, to=40,
                    increment=1, width=10, font=ENTRY_FONT).grid(row=r, column=1, **pad); r += 1

        # 螢幕與距離（cm）
        ttk.Label(self, text="Screen width (cm):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(self, textvariable=self.scr_width_cm_var, from_=10.0, to=300.0,
                    increment=0.5, width=10, font=ENTRY_FONT).grid(row=r, column=1, **pad); r += 1

        ttk.Label(self, text="Viewing distance (cm):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(self, textvariable=self.view_dist_cm_var, from_=10.0, to=300.0,
                    increment=1.0, width=10, font=ENTRY_FONT).grid(row=r, column=1, **pad); r += 1

        # 旋轉控制（可選）
        ttk.Checkbutton(self, text="Rotate stimulus (grating)", variable=self.rotate_var)\
            .grid(row=r, column=0, sticky="w", **pad); r += 1

        ttk.Label(self, text="Rotation Speed (deg/s):", font=LABEL_FONT)\
            .grid(row=r, column=0, sticky="w", **pad)
        self.rot_speed_spin = ttk.Spinbox(self, textvariable=self.rot_speed_var, from_=0, to=2000,
                                          increment=1, width=10, font=ENTRY_FONT)
        self.rot_speed_spin.grid(row=r, column=1, **pad); r += 1

        ttk.Label(self, text="Direction:", font=LABEL_FONT)\
            .grid(row=r, column=0, sticky="w", **pad)
        self.rot_dir_combo = ttk.Combobox(self, textvariable=self.rot_dir_var, values=["CW","CCW"],
                                          state="readonly", width=10, font=ENTRY_FONT)
        self.rot_dir_combo.grid(row=r, column=1, **pad); r += 1

        # 勾選時才啟用速度/方向
        def _toggle_rotate_controls(*args):
            state = "normal" if self.rotate_var.get() else "disabled"
            self.rot_speed_spin.configure(state=state)
            self.rot_dir_combo.configure(state=state)
        self.rotate_var.trace_add("write", _toggle_rotate_controls)
        _toggle_rotate_controls()

        # Staircase 參數（以 cpd）
        ttk.Label(self, text="Staircase start (cpd):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(self, textvariable=self.start_cpd_var, from_=0.1, to=60.0,
                    increment=0.1, width=10, font=ENTRY_FONT).grid(row=r, column=1, **pad); r += 1

        ttk.Label(self, text="Staircase step (cpd):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(self, textvariable=self.step_cpd_var, from_=0.05, to=10.0,
                    increment=0.05, width=10, font=ENTRY_FONT).grid(row=r, column=1, **pad); r += 1

        ttk.Label(self, text="Staircase min/max (cpd):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        frame_mm = ttk.Frame(self); frame_mm.grid(row=r, column=1, sticky="w", **pad)
        ttk.Spinbox(frame_mm, textvariable=self.min_cpd_var, from_=0.01, to=60.0,
                    increment=0.05, width=6, font=ENTRY_FONT).pack(side="left")
        ttk.Label(frame_mm, text=" ~ ", font=LABEL_FONT).pack(side="left")
        ttk.Spinbox(frame_mm, textvariable=self.max_cpd_var, from_=0.1, to=60.0,
                    increment=0.5, width=6, font=ENTRY_FONT).pack(side="left")
        r += 1

        ttk.Button(self, text="Start Test", command=self.on_start,
                   style="Big.TButton").grid(row=r, columnspan=3, pady=20)

        style = ttk.Style()
        style.configure("Big.TButton", font=("Arial", 16, "bold"), padding=10)

    def parse_rgb(self, s, default=(127,127,127)):
        try:
            parts = [int(x.strip()) for x in s.split(",")]
            if len(parts) == 3:
                return tuple(np.clip(parts, 0, 255))
        except:
            pass
        return default

    def on_start(self):
        sw_cm   = float(self.scr_width_cm_var.get())
        dist_cm = float(self.view_dist_cm_var.get())
        sw_deg  = screen_width_deg_from_cm(sw_cm, dist_cm)

        self.cfg = {
            'user_name' : self.user_var.get(),
            'calib_pts' : int(self.calib_var.get()),
            'stim_dur'  : float(self.stim_var.get()),
            'blank_dur' : float(self.blank_var.get()),
            'radius'    : int(self.rad_var.get()),
            'thresh'    : float(self.thresh_var.get()),
            'rotate'    : bool(self.rotate_var.get()),
            'rot_speed' : float(self.rot_speed_var.get()),
            'rot_dir'   : (1 if self.rot_dir_var.get() == "CW" else -1),
            'color_light': self.parse_rgb(self.color_light_var.get(), (255,255,255)),
            'color_dark' : self.parse_rgb(self.color_dark_var.get(),  (0,0,0)),
            'bg_color'   : self.parse_rgb(self.bg_color_var.get(),    (0,0,0)),
            # 幾何資訊
            'screen_width_cm'  : sw_cm,
            'view_distance_cm' : dist_cm,
            'screen_width_deg' : sw_deg,
            # Staircase（以 cpd）
            'start_cpd': float(self.start_cpd_var.get()),
            'step_cpd' : float(self.step_cpd_var.get()),
            'min_cpd'  : float(self.min_cpd_var.get()),
            'max_cpd'  : float(self.max_cpd_var.get()),
            # 校正圖
            'cali_img_path': self.cali_img_path_var.get().strip(),
            'cali_img_size': (int(self.cali_img_w_var.get()), int(self.cali_img_h_var.get())),
            # 視線標記
            'gaze_marker_color' : self.parse_rgb(self.gaze_color_var.get(), (0,255,0)),
            'gaze_marker_radius': int(self.gaze_radius_var.get()),
            'gaze_marker_width' : int(self.gaze_width_var.get()),
            # 外部 checkpoint 目錄（可選）
            'calib_dir': self.calib_dir_var.get().strip(),
        }
        self.destroy()

# ---------- 主實驗 ----------
def run_test(cfg):
    pygame.init()
    info = pygame.display.Info()
    W, H = info.current_w, info.current_h
    win = pygame.display.set_mode((W, H), pygame.FULLSCREEN)

    dcfg = DefaultConfig()
    dcfg.cali_mode = cfg['calib_pts']
    # 套用 GUI 選擇的校正圖片與尺寸（有選才覆蓋）
    if cfg.get('cali_img_path'):
        dcfg.cali_target_img = cfg['cali_img_path']
    if cfg.get('cali_img_size'):
        dcfg.cali_target_size = tuple(cfg['cali_img_size'])

    # ---- 校正 checkpoint 路徑決定：GUI 指定 > 預設 per-user/pt/res 資料夾
    if cfg.get('calib_dir'):
        profile_dir = Path(cfg['calib_dir'])
    else:
        profile_dir = _profile_dir(cfg, W, H)
    profile_dir.mkdir(parents=True, exist_ok=True)
    print("Calibration folder:", profile_dir)

    # 建立校正器並交給 GazeFollower
    calib = SVRCalibration(model_save_path=str(profile_dir))
    gf = GazeFollower(config=dcfg, calibration=calib)

    # 預覽 & 校正流程：有檔就跳過；沒檔就跑一次並儲存
    prep_input_for_calibration()
    ensure_pygame_focus()
    gf.preview(win=win)

    if not gf.calibration.has_calibrated:
        print("No checkpoint found → running calibration…")
        gf.calibrate(win=win)
        ok = gf.calibration.save_model()
        print("Calibration saved:", ok)
    else:
        print("Loaded checkpoint → skip calibrate()")
    restore_event_filter()

    gf.start_sampling()
    time.sleep(0.1)

    # Staircase 以 cpd
    stair = Staircase(start=cfg['start_cpd'],
                      step =cfg['step_cpd'],
                      minv =cfg['min_cpd'],
                      maxv =cfg['max_cpd'])

    centers = {'left': (W // 4, H // 2), 'right': (3 * W // 4, H // 2)}
    clock   = pygame.time.Clock()
    results = []

    # 預先建立背景（兩個平均色圓）
    def build_bg_surface(rad):
        other_color = to_rgb_tuple(mean_color_rgb(cfg['color_light'], cfg['color_dark']))
        surf = pygame.Surface((W, H))
        surf.fill(to_rgb_tuple(cfg['bg_color']))
        for pos in (centers['left'], centers['right']):
            pygame.draw.circle(surf, other_color, pos, rad)
        return surf

    while not stair.done():
        side  = random.choice(['left', 'right'])
        cpd   = float(stair.freq)               # 本 trial 的空間頻率（cpd）
        cs    = cpd * cfg['screen_width_deg']   # 對應 cycles/screen
        rad   = int(cfg['radius'])
        diam  = rad * 2

        bg_surface = build_bg_surface(rad)

        # 刺激貼圖準備（每 trial 建一次）
        xx_patch, yy_patch, circle_mask_patch = prepare_patch_grid(rad)
        circle_alpha = (circle_mask_patch.astype(np.uint8) * 255)  # (Hpatch,Wpatch)
        patch_surf = pygame.Surface((diam, diam), pygame.SRCALPHA)

        # 預刺激畫面
        win.blit(bg_surface, (0, 0))
        pygame.display.flip()

        start  = time.time()
        passed = False
        gaze_q = deque(maxlen=10)
        hold_start = None

        x0 = centers[side][0] - rad
        y0 = centers[side][1] - rad

        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                    gf.stop_sampling(); gf.save_data('VA_raw.csv')
                    gf.release(); pygame.quit(); sys.exit()

            t = time.time() - start

            # 背景（兩圓皆在）
            win.blit(bg_surface, (0, 0))

            # 旋轉角度（GUI 可選）
            if cfg['rotate']:
                angle = (t * cfg.get('rot_speed', 60.0) * cfg.get('rot_dir', 1)) % 360.0
            else:
                angle = 0.0

            # 本幀刺激貼圖（以 cycles/screen 驅動）
            patch_rgb = generate_grating_oriented_patch(
                cs, xx_patch, yy_patch, angle, w_total_px=W,
                color_dark=cfg['color_dark'],
                color_light=cfg['color_light'],
                do_blur=DO_BLUR
            )  # (diam, diam, 3), RGB

            # 寫入 patch_surf（RGB + alpha）
            px = pygame.surfarray.pixels3d(patch_surf)
            px[:] = patch_rgb.swapaxes(0, 1)  # (W,H,3)
            del px
            pa = pygame.surfarray.pixels_alpha(patch_surf)
            pa[:] = circle_alpha.T            # (W,H)
            del pa

            # 貼到畫面（覆蓋刺激側的平均色圓）
            win.blit(patch_surf, (x0, y0))

            # 讀取凝視與通過判定
            gi = gf.get_gaze_info()
            dist = float('inf')

            status_ok = False
            coords_raw = None
            coords_filt = None
            coords = None

            if gi is not None:
                status_ok  = bool(getattr(gi, 'status', False))
                coords_filt = getattr(gi, 'filtered_gaze_coordinates', None)
                coords_raw  = getattr(gi, 'gaze_coordinates', None)
                coords = coords_filt or coords_raw

            # 只接受「偵測有效 + 合理範圍」的樣本；否則丟棄並重置計時
            def _valid(c):
                if c is None: return False
                x, y = c
                if math.isnan(x) or math.isnan(y): return False
                # 合理邊界：允許一點點超出螢幕，但不要太離譜
                return (-W <= x <= 2*W) and (-H <= y <= 2*H)

            if status_ok and _valid(coords):
                gx, gy = map(int, coords)
                gaze_q.append((gx, gy))
                avgx = sum(p[0] for p in gaze_q) / len(gaze_q)
                avgy = sum(p[1] for p in gaze_q) / len(gaze_q)

                # Clamp 僅供顯示用
                out_of_bounds = not (0 <= avgx < W and 0 <= avgy < H)
                avgx = max(0, min(W-1, int(avgx)))
                avgy = max(0, min(H-1, int(avgy)))

                dist = np.hypot(avgx - centers[side][0], avgy - centers[side][1])

                if dist <= cfg['thresh']:
                    if hold_start is None:
                        hold_start = time.time()
                    elif time.time() - hold_start >= 0.8:
                        passed = True
                        break
                else:
                    hold_start = None

                pygame.draw.circle(
                    win,
                    to_rgb_tuple(cfg['gaze_marker_color']),
                    (int(avgx), int(avgy)),
                    int(cfg['gaze_marker_radius']),
                    int(cfg['gaze_marker_width'])
                )

                if out_of_bounds:
                    font_small = pygame.font.SysFont(None, 24)
                    oob_txt = font_small.render(f"gaze off-screen raw={tuple(map(int,coords))}", True, (255,100,100))
                    win.blit(oob_txt, (10, 40))
            else:
                # 無效樣本：清掉隊列與計時，避免上一筆殘值影響
                gaze_q.clear()
                hold_start = None
                font_small = pygame.font.SysFont(None, 24)
                msg = "no gaze sample" if not status_ok else f"drop invalid sample: {coords}"
                no_txt = font_small.render(msg, True, (255,100,100))
                win.blit(no_txt, (10, 40))



            # 狀態列：顯示 cpd 與 cycles/screen
            font = pygame.font.SysFont(None, 30)
            txt = font.render(f"{cpd:.2f} cpd  ({cs:.1f} cyc/screen)  t={t:.1f}s  d={dist:.1f}",
                              True, (255,255,255))
            win.blit(txt, (10, 10))

            pygame.display.flip()
            clock.tick(60)
            if t > cfg['stim_dur']:
                break

        # PASS / FAIL
        fb_font = pygame.font.SysFont(None, 100)
        fb_text = "PASS" if passed else "FAIL"
        color   = (0, 255, 0) if passed else (255, 0, 0)
        fb_surf = fb_font.render(fb_text, True, color)
        win.blit(fb_surf, ((W - fb_surf.get_width()) // 2, (H - fb_surf.get_height()) // 2))
        pygame.display.flip()
        time.sleep(1.0)

        stair.update(passed)
        results.append({'cpd': cpd, 'cycles_per_screen': cs, 'side': side, 'res': fb_text})
        time.sleep(cfg['blank_dur'])

    # ---- 結算（以 cpd）----
    final_cpd = float(stair.freq)
    va_score  = get_va_result_cpd(final_cpd, cfg['screen_width_deg'])
    for r in results:
        r['final_cpd'] = final_cpd
        r['va_score']  = va_score

    # 最終畫面
    result_font = pygame.font.SysFont(None, 80)
    info_font   = pygame.font.SysFont(None, 40)

    win.fill(cfg['bg_color'])
    text1 = result_font.render(f"Final Spatial Freq: {final_cpd:.2f} cpd", True, (255, 255, 255))
    text2 = result_font.render(f"Estimated VA Score: {va_score}", True, (0, 255, 255))
    text3 = info_font.render("Press Q to Exit", True, (200, 200, 200))

    win.blit(text1, ((W - text1.get_width()) // 2, H // 3 - 50))
    win.blit(text2, ((W - text2.get_width()) // 2, H // 3 + 50))
    win.blit(text3, ((W - text3.get_width()) // 2, H // 3 + 150))
    pygame.display.flip()

    # ====== 輸出 CSV ======
    import pandas as pd
    os.makedirs("VA_output", exist_ok=True)
    pd.DataFrame(results).to_csv(f"VA_output/VA_{cfg['user_name']}.csv", index=False, encoding='utf-8-sig')
    print(f"🔸 試次紀錄已輸出至  VA_output/VA_{cfg['user_name']}.csv")

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                pygame.quit(); sys.exit()
        time.sleep(0.05)

    # 不太會到這行，但保險
    gf.stop_sampling(); gf.release(); pygame.quit()

# ---------- main ----------
if __name__ == '__main__':
    os.makedirs("VA_output", exist_ok=True)
    logging.basicConfig(level=logging.DEBUG)

    # ---- 初始化 gazefollower 的 logger（要檔案路徑，不是資料夾）----
    from pathlib import Path
    import time, tempfile

    try:
        log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"gazefollower_{time.strftime('%Y%m%d_%H%M%S')}.log"
        GFLog.init(str(log_file))   # ✅ 傳入完整檔案路徑
    except Exception:
        # 若專案資料夾被鎖，退回到 temp 也行
        tmp = Path(tempfile.gettempdir()) / "GazeFollower" / "gazefollower.log"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        GFLog.init(str(tmp))

    s = SettingsWindow()
    s.mainloop()
    if s.cfg is None:
        print("User cancelled."); sys.exit(0)
    run_test(s.cfg)
