# -*- coding: utf-8 -*-
# ================================================================
# SVOP Demo ‧ 隱藏側邊按鈕版（可從 GUI 選擇Goldmann II~V、刺激圖片、並持續旋轉）
# 結束時顯示 PASS/FAIL 總數
# ================================================================
import os, sys, math, time, datetime, logging
from collections import deque
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pygame, pandas as pd

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
ANGULAR_DIAMETERS = {
    "Goldmann II":   0.43,
    "Goldmann III":  0.64,
    "Goldmann IV":   0.86,
    "Goldmann V":    1.72
}
SHOW_BUTTONS      = False
PASS_DWELL_SEC    = 2.0
TIMEOUT_SEC       = 5.0
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
        self.root.geometry("500x800")

        # User name
        tk.Label(self.root, text="User Name:").pack(pady=5)
        self.user_name = tk.StringVar(value="test_subject")
        tk.Entry(self.root, textvariable=self.user_name).pack()

        # Goldmann Size
        tk.Label(self.root, text="Goldmann Size:").pack(pady=5)
        self.size = tk.StringVar(value="Goldmann IV")
        ttk.Combobox(
            self.root,
            textvariable=self.size,
            values=list(ANGULAR_DIAMETERS.keys()),
            state="readonly"
        ).pack()

        # Calibration points
        tk.Label(self.root, text="Calibration Points:").pack(pady=5)
        self.calib_points = tk.IntVar(value=9)
        ttk.Combobox(
            self.root,
            textvariable=self.calib_points,
            values=[5, 9, 13],
            state="readonly"
        ).pack()

        # Stimulus points
        tk.Label(self.root, text="Stimulus Points:").pack(pady=5)
        self.stim_points = tk.IntVar(value=9)
        ttk.Combobox(
            self.root,
            textvariable=self.stim_points,
            values=[5, 9, 13],
            state="readonly"
        ).pack()

        # Screen width
        tk.Label(self.root, text="Screen Width (cm):").pack(pady=5)
        self.screen_width_cm = tk.DoubleVar(value=52.704)
        tk.Entry(self.root, textvariable=self.screen_width_cm).pack()

        # Viewing distance
        tk.Label(self.root, text="Viewing Distance (cm):").pack(pady=5)
        self.viewing_distance_cm = tk.DoubleVar(value=45.0)
        tk.Entry(self.root, textvariable=self.viewing_distance_cm).pack()

        # Pass threshold
        tk.Label(self.root, text="Pass Threshold (px):").pack(pady=5)
        self.threshold_dist = tk.IntVar(value=500)
        tk.Entry(self.root, textvariable=self.threshold_dist).pack()

        # Enable rotation?
        self.enable_rotation = tk.BooleanVar(value=False)
        tk.Checkbutton(
            self.root,
            text="Enable Continuous Rotation",
            variable=self.enable_rotation
        ).pack(pady=5)

        # Rotation speed
        tk.Label(self.root, text="Rotation Speed (°/s):").pack(pady=5)
        self.rot_speed = tk.DoubleVar(value=90.0)
        tk.Entry(self.root, textvariable=self.rot_speed).pack()

        # Stimulus image
        tk.Label(self.root, text="Stimulus Image:").pack(pady=5)
        self.stim_path = tk.StringVar(value="./VF-test/ball.jpg")
        frame = tk.Frame(self.root); frame.pack(fill="x", padx=10)
        tk.Entry(frame, textvariable=self.stim_path).pack(side="left", expand=True, fill="x")
        tk.Button(frame, text="Browse…", command=self.browse).pack(side="right")

        # Start
        tk.Button(self.root, text="Start Test", command=self.on_start).pack(pady=15)
        self.root.mainloop()

    def browse(self):
        f = filedialog.askopenfilename(
            title="Select stimulus image",
            filetypes=[("Image", "*.png *.jpg *.jpeg *.bmp"), ("All", "*.*")]
        )
        if f:
            self.stim_path.set(f)

    def on_start(self):
        try:
            if self.screen_width_cm.get() <= 0 or self.viewing_distance_cm.get() <= 0:
                raise ValueError("Screen width/distance must be > 0")
            if self.enable_rotation.get() and self.rot_speed.get() < 0:
                raise ValueError("Rotate speed must be ≥ 0")
            if not os.path.isfile(self.stim_path.get()):
                raise ValueError("Stimulus image not found")
        except Exception as e:
            messagebox.showerror("Input Error", str(e))
            return
        self.root.destroy()

    def get(self):
        return {
            "user_name":           self.user_name.get().strip().replace(" ", "_"),
            "goldmann_size":       self.size.get(),
            "calib_points":        self.calib_points.get(),
            "stim_points":         self.stim_points.get(),
            "screen_width_cm":     self.screen_width_cm.get(),
            "viewing_distance_cm": self.viewing_distance_cm.get(),
            "threshold_dist":      self.threshold_dist.get(),
            "enable_rotation":     self.enable_rotation.get(),
            "rot_speed":           self.rot_speed.get(),
            "stim_path":           self.stim_path.get()
        }


# ----------------------------------------------------------------
# 2. Helpers
# ----------------------------------------------------------------
def angular_to_pixel_diameter(angle_deg, dist_cm, px_per_cm):
    size_cm = 2 * dist_cm * math.tan(math.radians(angle_deg/2))
    return int(size_cm * px_per_cm)

def get_quadrant(x, y, cx, cy):
    if x > cx and y < cy: return 1
    if x < cx and y < cy: return 2
    if x < cx and y > cy: return 3
    if x > cx and y > cy: return 4
    return None

def generate_points(n, max_deg_horizon,max_deg_vertical_deg):
    if n == 5:
        return [(0,0),( max_deg_horizon, max_deg_vertical_deg),(-max_deg_horizon, max_deg_vertical_deg),( max_deg_horizon,-max_deg_vertical_deg),(-max_deg_horizon,-max_deg_vertical_deg)]       #可以測到半個螢幕視角
    if n == 9:
        return [(0, max_deg_vertical_deg),( max_deg_horizon, max_deg_vertical_deg),(-max_deg_horizon, max_deg_vertical_deg)
                ,( max_deg_horizon,-max_deg_vertical_deg),(-max_deg_horizon,-max_deg_vertical_deg),( max_deg_horizon,0),
                (-max_deg_horizon,0),(0,-max_deg_vertical_deg),(0,0)]   #可以測到整個螢幕視角 (9個螢幕角落點)
    if n == 13: #可以測到整個螢幕視角
        return [
            (0,0),( max_deg_horizon,0),(-max_deg_horizon,0),(0, max_deg_vertical_deg),(0,-max_deg_vertical_deg),
            ( max_deg_horizon, max_deg_vertical_deg),(-max_deg_horizon, max_deg_vertical_deg),( max_deg_horizon,-max_deg_vertical_deg),(-max_deg_horizon,-max_deg_vertical_deg),
            ( max_deg_horizon, max_deg_vertical_deg/2),(-max_deg_horizon, max_deg_vertical_deg/2),( max_deg_horizon,-max_deg_vertical_deg/2),(-max_deg_horizon,-max_deg_vertical_deg/2)
        ]
    raise ValueError("num_points must be 5/9/13")

def convert_positions_to_pixels(deg_pts, w, h, px_per_cm, dist_cm, diameter_px):
    d2p = lambda d: int(px_per_cm * math.tan(math.radians(d)) * dist_cm)
    raw = [(w//2 + d2p(x), h//2 - d2p(y)) for x,y in deg_pts]
    margin = diameter_px//2 + 10
    right  = w - margin
    return [(max(margin,min(x,right)),max(margin,min(y,h-margin))) for x,y in raw]


# ----------------------------------------------------------------
# 3. SVOP Test (with continuous rotation & end summary)
# ----------------------------------------------------------------
def svop_test(screen, stim_pts, stim_deg_list, diameter_px, gf, stim_img, font,small_font, cfg):
    W,H = screen.get_size()
    cx,cy = W//2,H//2
    threshold  = cfg['threshold_dist']
    rot_speed  = cfg['rot_speed']
    do_rotate  = cfg['enable_rotation']
    orig_stim  = stim_img.copy()
    angle      = 0.0

    results = []

    # 一一呈現刺激
    for idx, stim in enumerate(stim_pts, start=1):
        target_q   = get_quadrant(stim[0], stim[1], cx, cy)
        dwell_start= None
        passed     = False
        t0         = time.time()
        last_t     = t0

        while True:
            now  = time.time()
            dt   = now - last_t
            last_t = now
            elapsed = now - t0

            # 按 Q 強制結束
            for ev in pygame.event.get():
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                    gf.stop_sampling()
                    gf.release(); pygame.quit(); sys.exit()

            screen.fill(BACKGROUND_COLOR)

            if do_rotate:
                angle = (angle + rot_speed*dt) % 360
                disp = pygame.transform.rotate(orig_stim, angle)
                rect = disp.get_rect(center=stim)
            else:
                disp = orig_stim
                rect = orig_stim.get_rect(center=stim)

            screen.blit(disp, rect)

            # 顯示秒數
            elapsed_txt = small_font.render(f"Time: {elapsed:.1f}s", True, (255,255,255))
            # screen.blit(elapsed_txt, (10,10))
            # 顯示刺激點角度
            x_deg,y_deg = stim_deg_list[idx-1]
            angle_txt = small_font.render(f"Angle: {x_deg:.1f}°, {y_deg:.1f}°", True, (255,255,255))
            # screen.blit(angle_txt, (10, 10+elapsed_txt.get_height()+5))

            # 讀取 gaze
            gi = gf.get_gaze_info()
            if gi and getattr(gi,'status',False) and gi.filtered_gaze_coordinates:
                gx,gy = map(int, gi.filtered_gaze_coordinates)
                curr_q= get_quadrant(gx,gy,cx,cy)
                dist_px = math.hypot(stim[0]-gx, stim[1]-gy)

                if target_q is None or curr_q is None:
                    inside = (dist_px <= threshold)
                else:
                    inside = (curr_q == target_q)

                if inside:
                    dwell_start = dwell_start or now
                else:
                    dwell_start = None

                pygame.draw.circle(screen, PASS_COLOR, (gx,gy), 20,2)

                if dwell_start and (now-dwell_start)>=PASS_DWELL_SEC:
                    passed = True
                    break

            if elapsed > TIMEOUT_SEC:
                passed = False
                break

            pygame.display.flip()
            time.sleep(0.01)
        stim_ecc = math.hypot(x_deg, y_deg)
        # 記錄結果
        results.append({
            "user_name": cfg['user_name'],
            "stim_index": idx,
            "stim_x": stim[0], "stim_y": stim[1],
            "x_deg": stim_deg_list[idx-1][0],
            "y_deg": stim_deg_list[idx-1][1],
            "stim_ecc_deg": stim_ecc,
            "target_quadrant": target_q,
            "result": "PASS" if passed else "FAIL"
        })

        # 顯示單次 PASS/FAIL
        screen.fill(BACKGROUND_COLOR)
        msg = "PASS" if passed else "FAIL"
        col = PASS_COLOR if passed else ERROR_COLOR
        txt = font.render(msg, True, col)
        screen.blit(txt, (cx-txt.get_width()//2, cy-txt.get_height()//2))
        pygame.display.flip()
        time.sleep(1)

    # 結束後顯示總數
    pass_count = sum(1 for r in results if r['result']=="PASS")
    fail_count = sum(1 for r in results if r['result']=="FAIL")
    screen.fill(BACKGROUND_COLOR)
    summary1 = font.render(f"Total PASS: {pass_count}", True, PASS_COLOR)
    summary2 = font.render(f"Total FAIL: {fail_count}", True, ERROR_COLOR)
    screen.blit(summary1, (cx-summary1.get_width()//2, cy- summary1.get_height()))
    screen.blit(summary2, (cx-summary2.get_width()//2, cy+10))
    pygame.display.flip()
    time.sleep(3)

    # 儲存 CSV
    df = pd.DataFrame(results)
    fname = f"VF_output/svop_{cfg['user_name']}_{datetime.datetime.now():%Y%m%d%H%M%S}.csv"
    df.to_csv(fname, index=False)
    logging.info(f"Results saved to {fname}")


# ----------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------
def main():
    cfg = ConfigGUI().get()

    pygame.init(); pygame.font.init()
    small_font = pygame.font.SysFont(None, 24)

    font = pygame.font.SysFont(None, 72)

    info = pygame.display.Info()
    W,H = info.current_w, info.current_h
    px_per_cm = W / cfg['screen_width_cm']
    dist_cm   = cfg['viewing_distance_cm']
    half_w_cm = cfg['screen_width_cm']/2
    max_deg_horizon    = math.degrees(math.atan(half_w_cm/dist_cm))
    screen_height_cm = cfg['screen_width_cm'] * (info.current_h / info.current_w)
    half_h_cm = screen_height_cm / 2
    screen = pygame.display.set_mode((W,H), pygame.FULLSCREEN)
    max_deg_vertical = math.degrees(math.atan(half_h_cm/dist_cm))

    # load stimulus
    if not os.path.isfile(cfg['stim_path']):
        messagebox.showerror("File Error", f"Image not found:\n{cfg['stim_path']}")
        sys.exit(1)
    raw = pygame.image.load(cfg['stim_path'])
    stim_raw = raw.convert_alpha() if raw.get_alpha() else raw.convert()

    angle_deg  = ANGULAR_DIAMETERS[cfg['goldmann_size']]
    diameter_px= angular_to_pixel_diameter(angle_deg, dist_cm, px_per_cm)
    stim_img   = pygame.transform.scale(stim_raw, (diameter_px, diameter_px))

    dcfg = DefaultConfig(); dcfg.cali_mode = cfg['calib_points']
    gf = GazeFollower(config=dcfg)
    gf.preview(win=screen); gf.calibrate(win=screen)
    gf.start_sampling(); time.sleep(0.1)

    pts_deg = generate_points(cfg['stim_points'], max_deg_horizon,max_deg_vertical)
    pts_px  = convert_positions_to_pixels(pts_deg, W, H, px_per_cm, dist_cm, diameter_px)

    svop_test(screen, pts_px, pts_deg, diameter_px, gf, stim_img, font,small_font, cfg)

    gf.stop_sampling(); pygame.quit()

if __name__ == "__main__":
    os.makedirs("VF_output", exist_ok=True)
    main()
