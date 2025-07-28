# -*- coding: utf-8 -*-
# ================================================================
# SVOP Demo ‧ 隱藏側邊按鈕版（可從 GUI 選擇刺激圖片）
# ================================================================
# 變動重點
#   • 新增 PASS_DWELL_SEC = 0.5，必須連續注視 0.5 s 才算 PASS
#   • svop_test() 內新增 dwell_start 邏輯
#   • 其餘結構與前版相同，方便比對
# ---------------------------------------------------------------

# ----------------------------------------------------------------
# Patch：PyInstaller one-file 時註冊 DLL 目錄
# ----------------------------------------------------------------
import os, sys
if getattr(sys, 'frozen', False):
    meipass = sys._MEIPASS
    os.add_dll_directory(os.path.join(meipass, "mediapipe", "python"))
    os.add_dll_directory(meipass)

# ----------------------------------------------------------------
# Imports
# ----------------------------------------------------------------
import math, time, datetime, logging
from collections import deque
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pygame, cv2, numpy as np, pandas as pd

import gazefollower
gazefollower.logging = logging
import gazefollower.face_alignment.MediaPipeFaceAlignment as mpa
mpa.logging = logging
from gazefollower import GazeFollower
from gazefollower.misc import DefaultConfig

# ----------------------------------------------------------------
# Logging
# ----------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("svop_debug.log"), logging.StreamHandler()]
)

# ----------------------------------------------------------------
# Global constants
# ----------------------------------------------------------------
ANGULAR_DIAMETERS = {"Goldmann IV": 0.86}
SHOW_BUTTONS      = False          # True → 顯示側邊按鈕；False → 只用熱鍵
PASS_DWELL_SEC    = 0.5            # ★ 必須連續注視 ≥0.5 s 才 PASS
BACKGROUND_COLOR  = (0, 0, 0)
PASS_COLOR        = (0, 255, 0)
ERROR_COLOR       = (255, 0, 0)

# ----------------------------------------------------------------
# 1. Config GUI
# ----------------------------------------------------------------
class ConfigGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SVOP Test Configuration")
        self.root.geometry("400x420")
        tk.Label(self.root, text="User Name:").pack(pady=5)
        self.user_name = tk.StringVar(value="test_subject")
        tk.Entry(self.root, textvariable=self.user_name).pack()
        # Calibration points
        tk.Label(self.root, text="Calibration Points:").pack(pady=5)
        self.calib_points = tk.IntVar(value=9)
        ttk.Combobox(self.root, textvariable=self.calib_points,
                     values=[5, 9, 13], state="readonly").pack()

        # Stimulus points
        tk.Label(self.root, text="Stimulus Points:").pack(pady=5)
        self.stim_points = tk.IntVar(value=5)
        ttk.Combobox(self.root, textvariable=self.stim_points,
                     values=[5, 9, 13], state="readonly").pack()

        # Screen width
        tk.Label(self.root, text="Screen Width (cm):").pack(pady=5)
        self.screen_width_cm = tk.DoubleVar(value=52.704)
        tk.Entry(self.root, textvariable=self.screen_width_cm).pack()

        # Viewing distance
        tk.Label(self.root, text="Viewing Distance (cm):").pack(pady=5)
        self.viewing_distance_cm = tk.DoubleVar(value=45.0)
        tk.Entry(self.root, textvariable=self.viewing_distance_cm).pack()

        

        # Stimulus image
        tk.Label(self.root, text="Stimulus Image:").pack(pady=5)
        self.stim_path = tk.StringVar(value="./VF-test/ball.jpg")
        frame = tk.Frame(self.root); frame.pack(fill="x", padx=10)
        tk.Entry(frame, textvariable=self.stim_path).pack(
            side="left", expand=True, fill="x")
        tk.Button(frame, text="Browse…", command=self.browse).pack(side="right")

        tk.Button(self.root, text="Start Test", command=self.on_start).pack(pady=15)
        self.root.mainloop()

    def browse(self):
        f = filedialog.askopenfilename(
            title="Select stimulus image",
            filetypes=[("Image", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")])
        if f: self.stim_path.set(f)

    def on_start(self):
        try:
            if self.screen_width_cm.get() <= 0 or self.viewing_distance_cm.get() <= 0:
                raise ValueError("Screen width / distance 必須為正值")
            if not os.path.isfile(self.stim_path.get()):
                raise ValueError("找不到刺激圖片")
        except Exception as e:
            messagebox.showerror("Input Error", str(e)); return
        self.root.destroy()

    def get(self):
            return dict(
                user_name=self.user_name.get().strip().replace(" ", "_"),
                calib_points=self.calib_points.get(),
                stim_points=self.stim_points.get(),
                screen_width_cm=self.screen_width_cm.get(),
                viewing_distance_cm=self.viewing_distance_cm.get(),
                stim_path=self.stim_path.get()
            )

# ----------------------------------------------------------------
# 2. Helpers
# ----------------------------------------------------------------
def angular_to_pixel_diameter(angle_deg, dist_cm, px_per_cm):
    size_cm = 2 * dist_cm * math.tan(math.radians(angle_deg/2))
    return int(size_cm * px_per_cm)

def generate_points(n):
    if n == 5:
        return [(10, 10), (0, 0), (-10, 10), (10, -10), (-10, -10)]
    if n == 9:
        return [(0, 10), (10, 10), (-10, 10),
                (10, -10), (-10, -10),
                (0, 20), (20, 0), (-20, 0), (0, -20)]
    if n == 13:
        pts = [(0, 0),
               (10, 10), (-10, 10), (10, -10), (-10, -10),
               (0, 20), (20, 0), (-20, 0), (0, -20)]
        pts += [(15, 15), (-15, 15), (15, -15), (-15, -15)]
        return pts
    raise ValueError("num_points 必須 5/9/13")

def convert_positions_to_pixels(deg_pts, w, h, px_per_cm, dist_cm,
                                diameter_px, show_buttons=False, sidebar=140):
    d2p = lambda d: int(px_per_cm * math.tan(math.radians(d)) * dist_cm)
    raw = [(w//2 + d2p(x), h//2 - d2p(y)) for x, y in deg_pts]
    margin = diameter_px//2 + 10
    right = w - margin - (sidebar if show_buttons else 0)
    clamped=[]
    for x,y in raw:
        clamped.append((max(margin,min(x,right)),
                        max(margin,min(y,h-margin))))
    return clamped

def dist(p1,p2): return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

# ----------------------------------------------------------------
# 3. SVOP Test
# ----------------------------------------------------------------
def svop_test(screen, stim_pts, diameter_px, gf, stim_img, font, cfg):
    W,H = screen.get_size()
    if SHOW_BUTTONS:
        bar_w, bh, g = 140, 40, 10
        bx = W - bar_w
        btns = {n: pygame.Rect(bx+10, 150+i*(bh+g), bar_w-20, bh)
                for i,n in enumerate(["Pause","Skip","Retry","Quit"])}
    else:
        btns={}

    if not SHOW_BUTTONS:
        screen.fill((0,0,0))
        for i,t in enumerate(["P=Pause","S=Skip","R=Retry","Q=Quit"]):
            img=font.render(t,True,(255,255,0))
            screen.blit(img,(W//2-img.get_width()//2, H//2+i*30))
        pygame.display.flip(); time.sleep(2)

    pending=list(enumerate(stim_pts,1)); results=[]
    while pending:
        idx,pos=pending.pop(0)
        gaze_q=deque(maxlen=10)
        paused=False
        t0=time.time(); tot_pause=0; p_start=None
        skip=retry=quit=False
        dwell_start=None; d=float('inf')

        while True:
            now=time.time()
            el_raw=now-t0
            el= (p_start-t0-tot_pause) if paused else (el_raw-tot_pause)

            for ev in pygame.event.get():
                if ev.type==pygame.QUIT: pygame.quit(); sys.exit()
                if ev.type==pygame.KEYDOWN:
                    if ev.key==pygame.K_p:
                        paused=not paused
                        if paused: p_start=time.time()
                        else: tot_pause+=time.time()-p_start
                    elif ev.key==pygame.K_s: skip=True
                    elif ev.key==pygame.K_r: retry=True
                    elif ev.key==pygame.K_q: quit=True
                if SHOW_BUTTONS and ev.type==pygame.MOUSEBUTTONDOWN:
                    mx,my=ev.pos
                    for n,r in btns.items():
                        if r.collidepoint(mx,my):
                            if n=="Pause": paused=not paused
                            elif n=="Skip": skip=True
                            elif n=="Retry": retry=True
                            elif n=="Quit": quit=True
                            break
            if skip or retry or quit: break

            if paused:
                screen.fill(BACKGROUND_COLOR)
                txt=font.render("PAUSED",True,(255,255,0))
                screen.blit(txt,(W//2-txt.get_width()//2,H//2))
                pygame.display.flip(); time.sleep(0.1); continue

            screen.fill(BACKGROUND_COLOR)
            if SHOW_BUTTONS:
                pygame.draw.rect(screen,(50,50,50),(bx,0,bar_w,H))
                for n,r in btns.items():
                    c={'Pause':(100,100,100),'Skip':ERROR_COLOR,
                       'Retry':PASS_COLOR,'Quit':(150,50,50)}[n]
                    pygame.draw.rect(screen,c,r)
                    screen.blit(font.render(n,True,(0,0,0)),
                                (r.x+10,r.y+8))

            screen.blit(stim_img, stim_img.get_rect(center=pos))

            gi=gf.get_gaze_info()
            if gi and getattr(gi,'status',False) and gi.filtered_gaze_coordinates:
                gx,gy=map(int,gi.filtered_gaze_coordinates)
                gaze_q.append((gx,gy))
                avgx=sum(x for x,_ in gaze_q)/len(gaze_q)
                avgy=sum(y for _,y in gaze_q)/len(gaze_q)
                d=dist(pos,(avgx,avgy))
                pygame.draw.circle(screen,PASS_COLOR,(int(avgx),int(avgy)),30,4)

            # --- dwell 判定 ---
            inside = (gaze_q and d <= diameter_px*15)
            if inside:
                dwell_start = dwell_start or time.time()
            else:
                dwell_start = None

            if dwell_start and (time.time() - dwell_start) >= PASS_DWELL_SEC:
                screen.fill(BACKGROUND_COLOR)
                txt=font.render("PASS",True,PASS_COLOR)
                screen.blit(txt,(W//2-txt.get_width()//2,
                                 H//2-txt.get_height()//2))
                pygame.display.flip(); time.sleep(1)
                passed=True; break

            if el > 5:
                screen.fill(BACKGROUND_COLOR)
                txt=font.render("FAIL",True,ERROR_COLOR)
                screen.blit(txt,(W//2-txt.get_width()//2,
                                 H//2-txt.get_height()//2))
                pygame.display.flip(); time.sleep(1)
                passed=False; break

            # info
            info=[f"Stim {idx}/{len(stim_pts)}",
                  f"Time: {el:.1f}s",
                  f"Dist: {'--' if d==float('inf') else f'{d:.1f}'}px"]
            for i,t in enumerate(info):
                screen.blit(font.render(t,True,(255,255,255)),
                            (W-210,10+i*25))
            pygame.display.flip(); time.sleep(0.01)

        if quit: pygame.quit(); sys.exit()
        if skip: continue
        if retry: pending.insert(0,(idx,pos)); continue
        results.append(dict(stim_index=idx, stim_x=pos[0], stim_y=pos[1],
                            distance=d, result="PASS" if passed else "FAIL"))
        time.sleep(0.5)

    pd.DataFrame(results).to_csv(
        f"svop_results_{cfg['user_name']}.csv",
        index=False)
    logging.info("Results saved.")

# ----------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------
def main():
    cfg = ConfigGUI().get()

    pygame.init(); pygame.font.init()
    font=pygame.font.SysFont(None,28)
    info=pygame.display.Info()
    W,H=info.current_w, info.current_h
    PX_PER_CM = W / cfg['screen_width_cm']
    screen=pygame.display.set_mode((W,H), pygame.FULLSCREEN)

    try:
        raw=pygame.image.load(cfg['stim_path']).convert_alpha()
    except Exception as e:
        logging.error(e); pygame.quit(); sys.exit("Cannot open image")
    radius_px = angular_to_pixel_diameter(
        ANGULAR_DIAMETERS["Goldmann IV"],
        cfg['viewing_distance_cm'], PX_PER_CM)
    stim_img = pygame.transform.scale(raw,(radius_px*5,radius_px*5))

    dcfg = DefaultConfig(); dcfg.cali_mode = cfg['calib_points']
    gf = GazeFollower(config=dcfg)
    gf.preview(win=screen); gf.calibrate(win=screen)
    gf.start_sampling(); time.sleep(0.1)

    pts_deg = generate_points(cfg['stim_points'])
    pts_px = convert_positions_to_pixels(
        pts_deg, W, H, PX_PER_CM, cfg['viewing_distance_cm'],
        radius_px, show_buttons=SHOW_BUTTONS)

    svop_test(screen, pts_px, radius_px, gf, stim_img, font, cfg)

    gf.stop_sampling(); os.makedirs("data", exist_ok=True)
    gf.save_data(os.path.join("data","svop_demo.csv"))
    gf.release(); pygame.quit()

# ----------------------------------------------------------------
if __name__ == "__main__":
    main()
