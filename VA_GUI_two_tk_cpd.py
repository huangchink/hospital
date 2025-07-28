# -*- coding: utf-8 -*-
import logging
import gazefollower
gazefollower.logging = logging
import gazefollower.face_alignment.MediaPipeFaceAlignment as mpa
mpa.logging = logging

import cv2
import pygame
import numpy as np
import time
import sys
import math
import random
from collections import deque
import tkinter as tk
from tkinter import ttk, colorchooser
from gazefollower import GazeFollower
from gazefollower.misc import DefaultConfig

DO_BLUR    = True        # True: 輕微高斯模糊，減少鋸齒
BLUR_KSIZE = (5, 5)
BLUR_SIGMA = 1.0

# ---------- Utils ----------
def to_rgb_tuple(rgb_like):
    return tuple(int(v) for v in rgb_like)

def screen_width_deg_from_cm(width_cm: float, dist_cm: float) -> float:
    """由螢幕可視寬度與觀看距離換算『螢幕橫向視角(度)』"""
    if dist_cm <= 0 or width_cm <= 0:
        return 0.0
    return 2.0 * math.degrees(math.atan((width_cm / 2.0) / dist_cm))

# ---------- VA 分數對應（以 cpd 判斷） ----------
def get_va_result_cpd(cpd_value: float, screen_width_deg: float):
    """
from 耀元
    """
    if screen_width_deg <= 0:
        return "Unknown"
    thr_cs = [900,800, 700, 600, 500, 400, 300, 200, 100]
    scores = [0.9,0.8 ,0.7 ,0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    thr_cpd = [t / screen_width_deg for t in thr_cs]
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
        self.freq = float(start)  # 這裡的 freq 其實就是 cpd
        self.step = float(step)
        self.minv, self.maxv = float(minv), float(maxv)
        self.reversals = []
        self.last_correct = None
        self.correct_streak = 0
        self.max_correct_streak = 0

    def update(self, correct):
        if self.last_correct is not None and correct != self.last_correct:
            self.reversals.append(self.freq)
        self.last_correct = correct

        if correct:
            self.correct_streak += 1
            self.max_correct_streak = max(self.max_correct_streak, self.correct_streak)
        else:
            self.correct_streak = 0

        delta = self.step if correct else -self.step
        self.freq = min(self.maxv, max(self.minv, self.freq + delta))

    def done(self):
        return (len(self.reversals) >= 4) or \
               (self.freq >= self.maxv and self.max_correct_streak >= 3)

# ---------- GUI ----------
class SettingsWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VA Test Settings")
        self.resizable(False, False)
        self.geometry("820x820")

        LABEL_FONT = ("Arial", 14)
        ENTRY_FONT = ("Arial", 14)
        BTN_FONT   = ("Arial", 16, "bold")

        # 基本參數
        self.user_var = tk.StringVar(value="anonymous")   # ← ① 新增

        self.calib_var  = tk.StringVar(value="5")
        self.stim_var   = tk.DoubleVar(value=5.0)
        self.blank_var  = tk.DoubleVar(value=1.0)
        self.rad_var    = tk.IntVar(value=400)
        self.thresh_var = tk.DoubleVar(value=400)
        self.rotate_var = tk.BooleanVar(value=False)     # ← 旋轉可選可不選
        self.rot_speed_var = tk.DoubleVar(value=1.0)   # deg/s
        self.rot_dir_var   = tk.StringVar(value="CW")   # CW / CCW

        # 顏色
        self.color_light_var = tk.StringVar(value="255,255,255")
        self.color_dark_var  = tk.StringVar(value="0,0,0")
        self.bg_color_var    = tk.StringVar(value="0,0,0")

        # 螢幕與距離（cm）
        self.scr_width_cm_var = tk.DoubleVar(value=53.0)
        self.view_dist_cm_var = tk.DoubleVar(value=60.0)

        # Staircase（以 cpd）
        self.start_cpd_var = tk.DoubleVar(value=2.0)
        self.step_cpd_var  = tk.DoubleVar(value=2)
        self.min_cpd_var   = tk.DoubleVar(value=1)
        self.max_cpd_var   = tk.DoubleVar(value=20.0)

        self.cfg = None
        pad = {'padx': 10, 'pady': 8}

        # 版面
        r = 0
        ttk.Label(self, text="User name:", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.user_var, font=ENTRY_FONT, width=15).grid(row=r, column=1, **pad)
        r += 1
        ttk.Label(self, text="Calibration Points:", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Combobox(self, textvariable=self.calib_var,
                     values=["5", "9", "13"], state="readonly", width=10, font=ENTRY_FONT)\
            .grid(row=r, column=1, **pad)
        r += 1

        ttk.Label(self, text="Stimulus Duration (s):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(self, textvariable=self.stim_var, from_=0.5, to=10.0,
                    increment=0.1, width=10, font=ENTRY_FONT)\
            .grid(row=r, column=1, **pad)
        r += 1

        ttk.Label(self, text="Blank Duration (s):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(self, textvariable=self.blank_var, from_=0.5, to=10.0,
                    increment=0.1, width=10, font=ENTRY_FONT)\
            .grid(row=r, column=1, **pad)
        r += 1

        ttk.Label(self, text="Circle Radius (px):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(self, textvariable=self.rad_var, from_=50, to=800,
                    increment=10, width=10, font=ENTRY_FONT)\
            .grid(row=r, column=1, **pad)
        r += 1

        ttk.Label(self, text="Threshold (px):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(self, textvariable=self.thresh_var, from_=10, to=1000,
                    increment=10, width=10, font=ENTRY_FONT)\
            .grid(row=r, column=1, **pad)
        r += 1

        def choose_color(target_var):
            color = colorchooser.askcolor()[0]
            if color:
                r_,g_,b_ = [int(c) for c in color]
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

        # 螢幕與距離（cm）
        ttk.Label(self, text="Screen width (cm):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(self, textvariable=self.scr_width_cm_var, from_=10.0, to=300.0,
                    increment=0.5, width=10, font=ENTRY_FONT).grid(row=r, column=1, **pad)
        r += 1

        ttk.Label(self, text="Viewing distance (cm):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(self, textvariable=self.view_dist_cm_var, from_=10.0, to=300.0,
                    increment=1.0, width=10, font=ENTRY_FONT).grid(row=r, column=1, **pad)
        r += 1

        # 旋轉控制（可選）
        ttk.Checkbutton(self, text="Rotate stimulus (grating)", variable=self.rotate_var)\
            .grid(row=r, column=0, sticky="w", **pad)
        r += 1  # ← 重要：換到下一列再放速度/方向

        ttk.Label(self, text="Rotation Speed (deg/s):", font=LABEL_FONT)\
            .grid(row=r, column=0, sticky="w", **pad)
        self.rot_speed_spin = ttk.Spinbox(self, textvariable=self.rot_speed_var, from_=0, to=2000,
                                        increment=1, width=10, font=ENTRY_FONT)
        self.rot_speed_spin.grid(row=r, column=1, **pad)
        r += 1

        ttk.Label(self, text="Direction:", font=LABEL_FONT)\
            .grid(row=r, column=0, sticky="w", **pad)
        self.rot_dir_combo = ttk.Combobox(self, textvariable=self.rot_dir_var, values=["CW","CCW"],
                                        state="readonly", width=10, font=ENTRY_FONT)
        self.rot_dir_combo.grid(row=r, column=1, **pad)
        r += 1

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
                    increment=0.1, width=10, font=ENTRY_FONT).grid(row=r, column=1, **pad)
        r += 1

        ttk.Label(self, text="Staircase step (cpd):", font=LABEL_FONT).grid(row=r, sticky="w", **pad)
        ttk.Spinbox(self, textvariable=self.step_cpd_var, from_=0.05, to=10.0,
                    increment=0.05, width=10, font=ENTRY_FONT).grid(row=r, column=1, **pad)
        r += 1

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
            if len(parts)==3:
                return tuple(np.clip(parts,0,255))
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
            'rotate'    : bool(self.rotate_var.get()),          # ← GUI 可選
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
    gf = GazeFollower(config=dcfg)
    gf.preview(win=win)
    gf.calibrate(win=win)
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
            if gi and getattr(gi, 'status', False) and gi.filtered_gaze_coordinates:
                gx, gy = map(int, gi.filtered_gaze_coordinates)
                gaze_q.append((gx, gy))
                avgx = sum(p[0] for p in gaze_q) / len(gaze_q)
                avgy = sum(p[1] for p in gaze_q) / len(gaze_q)
                dist = np.hypot(avgx - centers[side][0], avgy - centers[side][1])

                if dist <= cfg['thresh']:
                    if hold_start is None:
                        hold_start = time.time()
                    elif time.time() - hold_start >= 0.8:   # 持續 0.8s 視為 PASS
                        passed = True
                        break
                else:
                    hold_start = None

                # ★★★ 補回：畫出綠色凝視標記（滑動平均位置） ★★★
                if gaze_q:
                    pygame.draw.circle(win, (0, 255, 0), (int(avgx), int(avgy)), 30, 4)

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
# ---- 結算（以 cpd）----
    final_cpd = float(stair.freq)
    va_score  = get_va_result_cpd(final_cpd, cfg['screen_width_deg'])
    for r in results:
        r['final_cpd'] = final_cpd    # 每列都附上最終 cpd
        r['va_score']  = va_score     # 以及換算後的 VA 分數

    # ====== ①  把結果寫成 CSV  ======
    import pandas as pd


    pd.DataFrame(results).to_csv(f'VA_{cfg['user_name']}.csv', index=False, encoding='utf-8-sig')
    print(f'🔸 試次紀錄已輸出至 VA_{cfg['user_name']}.csv')
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                pygame.quit(); sys.exit()
        time.sleep(0.05)

    gf.stop_sampling(); gf.release(); pygame.quit()

# ---------- main ----------
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    s = SettingsWindow()
    s.mainloop()
    if s.cfg is None:
        print("User cancelled."); sys.exit(0)
    run_test(s.cfg)
