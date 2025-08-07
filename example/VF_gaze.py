import cv2
import pygame
import numpy as np
import pandas as pd
import math
import time
import datetime
import os
import sys
import logging
from collections import deque
from gazefollower import GazeFollower

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("svop_debug.log"),
        logging.StreamHandler()
    ]
)

# Constants
VIEWING_DISTANCE_CM = 45
SCREEN_WIDTH_CM = 52.704  # 螢幕寬度 (cm)
BACKGROUND_COLOR = (0, 0, 0)
PASS_COLOR = (0, 255, 0)
ERROR_COLOR = (255, 0, 0)
ANGULAR_DIAMETERS = {"Goldmann IV": 0.86}
THRESHOLD_DIST = 100  # pass 閾值 (px)

# Helper functions
def angular_to_pixel_diameter(angle_deg, dist_cm, px_per_cm):
    size_cm = 2 * dist_cm * math.tan(math.radians(angle_deg / 2))
    return int(size_cm * px_per_cm)

def generate_points(num_points):
    if num_points == 5:
        return [(0,0),(10,10),(-10,10),(10,-10),(-10,-10)]
    if num_points == 9:
        return [(0,10),(10,10),(-10,10),(10,-10),(-10,-10),(0,20),(20,0),(-20,0),(0,-20)]
    if num_points == 13:
        pts = [(0,0),(10,10),(-10,10),(10,-10),(-10,-10),(0,20),(20,0),(-20,0),(0,-20)]
        pts += [(15,15),(-15,15),(15,-15),(-15,-15)]
        return pts
    return []


def convert_positions_to_pixels(positions, width, height, px_per_cm):
    DEG_TO_PX = lambda deg: int(px_per_cm * math.tan(math.radians(deg)) * VIEWING_DISTANCE_CM)
    pixels = [(width // 2 + DEG_TO_PX(x), height // 2 - DEG_TO_PX(y)) for x, y in positions]
    return [(min(max(x, 0), width - 1), min(max(y, 0), height - 1)) for x, y in pixels]

def calculate_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# Main SVOP test

def svop_test(screen, stimulus_positions, diameter_px, gf, stim_img, font):
    data = []
    for stim in stimulus_positions:
        # 顯示刺激
        screen.fill(BACKGROUND_COLOR)
        rect = stim_img.get_rect(center=stim)
        screen.blit(stim_img, rect)
        pygame.display.flip()
        logging.info(f"Displaying stimulus at {stim}")

        start_time = time.time()
        passed = False
        gaze_queue = deque(maxlen=10)

        while True:
            # 處理 Pygame 事件
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
            pygame.event.pump()

            elapsed = time.time() - start_time
            if elapsed > 10:
                logging.warning(f"Timeout (10s) for stimulus {stim}")
                break

            # 嘗試取得 gaze
            gaze_info = gf.get_gaze_info()
            if (gaze_info is None or
                not getattr(gaze_info, 'status', False) or
                not hasattr(gaze_info, 'filtered_gaze_coordinates') or
                gaze_info.filtered_gaze_coordinates is None or
                len(gaze_info.filtered_gaze_coordinates) < 2):
                time.sleep(0.01)
                continue

            # 安全解包
            gx, gy = map(int, gaze_info.filtered_gaze_coordinates)
            gaze_queue.append((gx, gy))

            # 計算平均 gaze
            avg_x = sum(x for x, _ in gaze_queue) / len(gaze_queue)
            avg_y = sum(y for _, y in gaze_queue) / len(gaze_queue)
            dist = calculate_distance(stim, (avg_x, avg_y))

            # 重畫刺激與 gaze
            screen.fill(BACKGROUND_COLOR)
            screen.blit(stim_img, rect)
            pygame.draw.circle(screen, PASS_COLOR, (gx, gy), 20, 2)
            pygame.draw.circle(screen, PASS_COLOR, (int(avg_x), int(avg_y)), 30, 4)

            # PASS 判定
            if elapsed <= 2 and dist <= THRESHOLD_DIST:
                passed = True
                text = font.render("PASS", True, PASS_COLOR)
                screen.blit(text, (10, screen.get_height() - 40))
                pygame.display.flip()
                logging.info(f"PASS at {elapsed:.2f}s dist={dist}")
                data.append({
                    "stim_x": stim[0],
                    "stim_y": stim[1],
                    "gaze_x": avg_x,
                    "gaze_y": avg_y,
                    "distance": dist,
                    "result": "PASS"
                })
                time.sleep(1)
                break

            pygame.display.flip()
            time.sleep(0.01)

        # 若未 PASS，使用最後樣本記錄結果
        if not passed:
            if gaze_queue:
                avg_x = sum(x for x, _ in gaze_queue) / len(gaze_queue)
                avg_y = sum(y for _, y in gaze_queue) / len(gaze_queue)
                dist = calculate_distance(stim, (avg_x, avg_y))
                result = "PASS" if dist <= THRESHOLD_DIST else "FAIL"
                data.append({
                    "stim_x": stim[0],
                    "stim_y": stim[1],
                    "gaze_x": avg_x,
                    "gaze_y": avg_y,
                    "distance": dist,
                    "result": result
                })
                logging.info(f"Final sample dist={dist}, result={result}")
            else:
                data.append({
                    "stim_x": stim[0],
                    "stim_y": stim[1],
                    "gaze_x": None,
                    "gaze_y": None,
                    "distance": None,
                    "result": "FAIL"
                })
                logging.warning("No gaze data collected for stimulus {stim}")
            time.sleep(1)

    # 儲存結果
    df = pd.DataFrame(data)
    fname = f"svop_results_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    df.to_csv(fname, index=False)
    logging.info(f"Results saved to {fname}")


if __name__ == '__main__':
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont(None, 36)

    info = pygame.display.Info()
    WIDTH, HEIGHT = info.current_w, info.current_h
    PX_PER_CM = WIDTH / SCREEN_WIDTH_CM
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)

    # 載入並縮放刺激圖片
    stim_img = pygame.image.load("./VF-test/ball.jpg").convert_alpha()
    diameter_px = angular_to_pixel_diameter(ANGULAR_DIAMETERS['Goldmann IV'], VIEWING_DISTANCE_CM, PX_PER_CM)
    stim_img = pygame.transform.scale(stim_img, (diameter_px*2, diameter_px*2))

    # 初始化 gaze follower 並校正 5 點
    from gazefollower.misc import DefaultConfig
    cfg = DefaultConfig()
    cfg.cali_mode = 5
    gf = GazeFollower(config=cfg)
    gf.preview(win=screen)
    gf.calibrate(win=screen)
    gf.start_sampling()
    time.sleep(0.1)

    # 準備刺激點並執行測試
    pts_deg = generate_points(9)
    stim_px = convert_positions_to_pixels(pts_deg, WIDTH, HEIGHT, PX_PER_CM)
    svop_test(screen, stim_px, diameter_px, gf, stim_img, font)

    # 清理與儲存原始 gaze data
    time.sleep(0.1)
    gf.stop_sampling()
    os.makedirs('data', exist_ok=True)
    gf.save_data(os.path.join('data', 'svop_demo.csv'))
    gf.release()
    pygame.quit()
