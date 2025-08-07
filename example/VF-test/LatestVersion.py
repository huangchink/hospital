import cv2
import pygame
import numpy as np
import pandas as pd
import mediapipe as mp
import math
import time
import datetime
from scipy.signal import savgol_filter
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import sys
import logging

# 設置日誌
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("svop_debug.log"),
        logging.StreamHandler()
    ]
)

# 初始化 Pygame
pygame.init()

# 螢幕和顯示參數
VIEWING_DISTANCE_CM = 45
SCREEN_WIDTH_CM = 47.2
BACKGROUND_COLOR = (0, 0, 0)
STIMULUS_COLOR = (255, 255, 255)

# Goldmann 刺激物的角直徑 (度)
ANGULAR_DIAMETERS = {
    "Goldmann III": 0.43,
    "Goldmann IV": 0.86,
    "Goldmann V": 1.72
}

# Mediapipe 設定
mp_face_mesh = mp.solutions.face_mesh

# 處理打包後的資源路徑
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# GUI 類別
class SVOPSettingsGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("SVOP Test Settings")
        self.master.geometry("450x450")

        tk.Label(master, text="Calibration Points Number:").pack(pady=5)
        self.calib_points_var = tk.IntVar(value=9)
        ttk.Combobox(master, textvariable=self.calib_points_var, values=[5, 9, 13], state="readonly").pack()

        tk.Label(master, text="Calibration Point Speed (seconds/point):").pack(pady=5)
        self.calib_speed_var = tk.DoubleVar(value=10.0)
        tk.Entry(master, textvariable=self.calib_speed_var).pack()

        tk.Label(master, text="Calibration Point Image:").pack(pady=5)
        self.calib_image_path_var = tk.StringVar(value="")
        self.image_path_label = tk.Label(master, text="No image selected", wraplength=400)
        self.image_path_label.pack()
        tk.Button(master, text="Select Image", command=self.select_image).pack(pady=5)

        tk.Label(master, text="Stimuli Points Number:").pack(pady=5)
        self.stimuli_points_var = tk.IntVar(value=9)
        ttk.Combobox(master, textvariable=self.stimuli_points_var, values=[5, 9, 13], state="readonly").pack()

        tk.Label(master, text="Camera Index:").pack(pady=5)
        self.camera_index_var = tk.IntVar(value=0)
        ttk.Combobox(master, textvariable=self.camera_index_var, values=[0, 1, 2], state="readonly").pack()

        tk.Button(master, text="Start Test", command=self.start_test).pack(pady=20)

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if file_path:
            self.calib_image_path_var.set(file_path)
            self.image_path_label.config(text=file_path)

    def start_test(self):
        try:
            calib_points = self.calib_points_var.get()
            calib_speed = self.calib_speed_var.get()
            calib_image_path = self.calib_image_path_var.get()
            stimuli_points = self.stimuli_points_var.get()
            camera_index = self.camera_index_var.get()

            if calib_speed <= 0:
                raise ValueError("Calibration speed must be positive")
            if calib_points < 5 or stimuli_points < 5:
                raise ValueError("Points number must be at least 5")
            if not calib_image_path or not os.path.exists(calib_image_path):
                raise ValueError("Please select a valid image file")

            logging.info(f"Starting test with parameters: {calib_points}, {calib_speed}, {calib_image_path}, {stimuli_points}, {camera_index}")
            self.master.destroy()
            main_test(calib_points, calib_speed, calib_image_path, stimuli_points, camera_index)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start test: {str(e)}")
            logging.error(f"Start test error: {str(e)}")

def angular_to_pixel_diameter(angular_diameter, distance_cm, pixels_per_cm):
    size_cm = 2 * distance_cm * math.tan(math.radians(angular_diameter / 2))
    return int(size_cm * pixels_per_cm)

def convert_positions_to_pixels(positions, screen_width, screen_height):
    DEG_TO_PIXEL = lambda deg: int(PIXELS_PER_CM * math.tan(math.radians(deg)) * VIEWING_DISTANCE_CM)
    pixel_positions = [
        (screen_width // 2 + DEG_TO_PIXEL(x), screen_height // 2 - DEG_TO_PIXEL(y))
        for x, y in positions
    ]
    pixel_positions = [
        (min(max(x, 0), screen_width - 1),
         min(max(y, 20), screen_height - 20))
        for x, y in pixel_positions
    ]
    return pixel_positions

def get_gaze_position(frame, face_landmarks):
    try:
        if face_landmarks:
            left_eye_indices = [33, 133, 160, 158, 153, 144, 362, 249]
            right_eye_indices = [362, 263, 466, 469, 470, 471, 472, 473]
            left_eye_coords = [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in left_eye_indices]
            right_eye_coords = [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in right_eye_indices]
            left_eye_center = np.mean(left_eye_coords, axis=0)
            right_eye_center = np.mean(right_eye_coords, axis=0)
            eye_center = np.mean([left_eye_center, right_eye_center], axis=0)
            gaze_x = eye_center[0] * SCREEN_WIDTH
            gaze_y = eye_center[1] * SCREEN_HEIGHT
            if np.isnan(gaze_x) or np.isnan(gaze_y):
                logging.warning("Invalid gaze coordinates detected (NaN)")
                return None
            return int(gaze_x), int(gaze_y)
        return None
    except Exception as e:
        logging.error(f"Error in get_gaze_position: {str(e)}")
        return None

def smooth_gaze_data(gaze_points, window_size=5, poly_order=2):
    try:
        if len(gaze_points) < window_size:
            logging.debug(f"Not enough gaze points for smoothing: {len(gaze_points)}")
            return gaze_points
        x_coords = [p[0] for p in gaze_points]
        y_coords = [p[1] for p in gaze_points]
        if any(np.isnan(x) for x in x_coords) or any(np.isnan(y) for y in y_coords):
            logging.warning("NaN values found in gaze points, skipping smoothing")
            return gaze_points
        smoothed_x = savgol_filter(x_coords, window_size, poly_order)
        smoothed_y = savgol_filter(y_coords, window_size, poly_order)
        return list(zip(smoothed_x, smoothed_y))
    except Exception as e:
        logging.error(f"Error in smooth_gaze_data: {str(e)}")
        return gaze_points

def remove_outliers(data, z_threshold=3):
    try:
        if len(data) == 0:
            return data
        data = np.array(data)
        if data.size == 0 or np.any(np.isnan(data)):
            logging.warning("Invalid data in remove_outliers (empty or NaN)")
            return data
        z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
        return data[(z_scores < z_threshold).all(axis=1)]
    except Exception as e:
        logging.error(f"Error in remove_outliers: {str(e)}")
        return data

def generate_points(num_points):
    if num_points == 5:
        return [(0, 0), (10, 10), (-10, 10), (10, -10), (-10, -10)]
    elif num_points == 9:
        return [(0, 10), (10, 10), (-10, 10), (10, -10), (-10, -10), (0, 20), (20, 0), (-20, 0), (0, -20)]
    elif num_points == 13:
        return [(0, 0), (10, 10), (-10, 10), (10, -10), (-10, -10), 
                (0, 20), (20, 0), (-20, 0), (0, -20), (15, 15), (-15, 15), (15, -15), (-15, -15)]
    return []

def calibration_phase(screen, cap, calib_points, calib_speed, calib_image_path):
    logging.info("Starting calibration...")
    calibration_positions = convert_positions_to_pixels(generate_points(calib_points), SCREEN_WIDTH, SCREEN_HEIGHT)
    raw_gaze_points = []

    try:
        calib_image_path = resource_path(calib_image_path)
        if not os.path.exists(calib_image_path):
            raise FileNotFoundError(f"Calibration image not found at {calib_image_path}")
        calib_image = pygame.image.load(calib_image_path)
        calib_image = pygame.transform.scale(calib_image, (40, 40))
    except Exception as e:
        logging.error(f"Error loading calibration image: {e}")
        pygame.quit()
        sys.exit(1)

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        for point in calibration_positions:
            screen.fill(BACKGROUND_COLOR)
            image_rect = calib_image.get_rect(center=point)
            screen.blit(calib_image, image_rect)
            pygame.display.flip()
            logging.info(f"Displaying calibration point at {point}")

            gaze_data = []
            start_time = time.time()
            while time.time() - start_time < calib_speed:
                try:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        logging.error("Failed to capture frame from camera")
                        break
                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb_frame)
                    if results.multi_face_landmarks:
                        gaze = get_gaze_position(frame, results.multi_face_landmarks[0])
                        if gaze:
                            gaze_data.append(gaze)
                            logging.debug(f"Gaze point collected: {gaze}")
                    else:
                        logging.debug("No face landmarks detected")
                except Exception as e:
                    logging.error(f"Error processing frame in calibration: {str(e)}")
                    break

            logging.info(f"Processing gaze data for point {point}, collected {len(gaze_data)} points")
            if len(gaze_data) < 10:  # 確保有足夠數據點
                logging.warning(f"Insufficient gaze data for point {point}: {len(gaze_data)} points")
                raw_gaze_points.append(point)  # 使用預設值
                continue

            try:
                gaze_data = remove_outliers(gaze_data)
                logging.debug(f"After outlier removal: {len(gaze_data)} points remain")
                if len(gaze_data) < 5:
                    logging.warning(f"Too few points after outlier removal for point {point}")
                    raw_gaze_points.append(point)
                    continue

                smoothed_gaze = smooth_gaze_data(gaze_data)
                logging.debug(f"Smoothed gaze data length: {len(smoothed_gaze)}")
                if not smoothed_gaze or len(smoothed_gaze) == 0:
                    logging.warning(f"Failed to smooth gaze data for point {point}")
                    raw_gaze_points.append(point)
                    continue

                avg_gaze = np.mean(smoothed_gaze, axis=0)
                if np.any(np.isnan(avg_gaze)) or np.any(np.isinf(avg_gaze)):
                    logging.warning(f"Invalid average gaze data for point {point}: {avg_gaze}")
                    raw_gaze_points.append(point)
                else:
                    raw_gaze_points.append(avg_gaze)
                    logging.info(f"Gaze data collected for point {point}: {avg_gaze}")
            except Exception as e:
                logging.error(f"Error processing gaze data for point {point}: {str(e)}")
                raw_gaze_points.append(point)  # 發生錯誤時使用校準點作為預設

    if len(raw_gaze_points) < calib_points:
        logging.error(f"Calibration failed: Only {len(raw_gaze_points)}/{calib_points} gaze points collected.")
        pygame.quit()
        sys.exit(1)

    raw_gaze_points = np.array(raw_gaze_points)
    screen_points = np.array(calibration_positions)
    logging.debug(f"Raw gaze points: {raw_gaze_points}")
    logging.debug(f"Screen points: {screen_points}")

    try:
        logging.info("Fitting calibration model...")
        x_model = np.poly1d(np.polyfit(raw_gaze_points[:, 0], screen_points[:, 0], 3))
        y_model = np.poly1d(np.polyfit(raw_gaze_points[:, 1], screen_points[:, 1], 3))
        logging.info("Calibration model fitted successfully")
    except Exception as e:
        logging.error(f"Error fitting calibration model: {str(e)}")
        pygame.quit()
        sys.exit(1)

    logging.info("Calibration complete!")
    return x_model, y_model

def remove_outliers(data, z_threshold=3):
    try:
        if len(data) < 5:
            logging.debug("Not enough data points for outlier removal")
            return data
        data = np.array(data)
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            logging.warning("NaN or Inf values found in gaze data")
            return data
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        if np.any(std == 0):
            logging.warning("Zero standard deviation detected, skipping outlier removal")
            return data
        z_scores = np.abs((data - mean) / std)
        filtered_data = data[(z_scores < z_threshold).all(axis=1)]
        logging.debug(f"Outlier removal: {len(data)} -> {len(filtered_data)} points")
        return filtered_data if len(filtered_data) > 0 else data
    except Exception as e:
        logging.error(f"Error in remove_outliers: {str(e)}")
        return data

def smooth_gaze_data(gaze_points, window_size=5, poly_order=2):
    try:
        if len(gaze_points) < window_size:
            logging.debug(f"Not enough gaze points for smoothing: {len(gaze_points)}")
            return gaze_points
        x_coords = [p[0] for p in gaze_points]
        y_coords = [p[1] for p in gaze_points]
        if any(np.isnan(x) or np.isinf(x) for x in x_coords) or any(np.isnan(y) or np.isinf(y) for y in y_coords):
            logging.warning("NaN or Inf values in gaze points, skipping smoothing")
            return gaze_points
        window_size = min(window_size, len(gaze_points) // 2 * 2 + 1)  # 確保窗口大小為奇數且不超過數據長度
        smoothed_x = savgol_filter(x_coords, window_size, poly_order)
        smoothed_y = savgol_filter(y_coords, window_size, poly_order)
        return list(zip(smoothed_x, smoothed_y))
    except Exception as e:
        logging.error(f"Error in smooth_gaze_data: {str(e)}")
        return gaze_points

def remove_outliers(data, z_threshold=3):
    try:
        if len(data) < 5:
            logging.debug("Not enough data points for outlier removal")
            return data
        data = np.array(data)
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            logging.warning("NaN or Inf values found in gaze data")
            return data
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        if np.any(std == 0):
            logging.warning("Zero standard deviation detected, skipping outlier removal")
            return data
        z_scores = np.abs((data - mean) / std)
        filtered_data = data[(z_scores < z_threshold).all(axis=1)]
        logging.debug(f"Outlier removal: {len(data)} -> {len(filtered_data)} points")
        return filtered_data if len(filtered_data) > 0 else data
    except Exception as e:
        logging.error(f"Error in remove_outliers: {str(e)}")
        return data

def smooth_gaze_data(gaze_points, window_size=5, poly_order=2):
    try:
        if len(gaze_points) < window_size:
            logging.debug(f"Not enough gaze points for smoothing: {len(gaze_points)}")
            return gaze_points
        x_coords = [p[0] for p in gaze_points]
        y_coords = [p[1] for p in gaze_points]
        if any(np.isnan(x) or np.isinf(x) for x in x_coords) or any(np.isnan(y) or np.isinf(y) for y in y_coords):
            logging.warning("NaN or Inf values in gaze points, skipping smoothing")
            return gaze_points
        window_size = min(window_size, len(gaze_points) // 2 * 2 + 1)  # 確保窗口大小為奇數且不超過數據長度
        smoothed_x = savgol_filter(x_coords, window_size, poly_order)
        smoothed_y = savgol_filter(y_coords, window_size, poly_order)
        return list(zip(smoothed_x, smoothed_y))
    except Exception as e:
        logging.error(f"Error in smooth_gaze_data: {str(e)}")
        return gaze_points
    logging.info("Starting calibration...")
    calibration_positions = convert_positions_to_pixels(generate_points(calib_points), SCREEN_WIDTH, SCREEN_HEIGHT)
    raw_gaze_points = []

    try:
        calib_image_path = resource_path(calib_image_path)
        if not os.path.exists(calib_image_path):
            raise FileNotFoundError(f"Calibration image not found at {calib_image_path}")
        calib_image = pygame.image.load(calib_image_path)
        calib_image = pygame.transform.scale(calib_image, (40, 40))
    except FileNotFoundError as e:
        logging.error(f"Error loading calibration image: {e}")
        pygame.quit()
        sys.exit(1)
    except pygame.error as e:
        logging.error(f"Pygame error loading image: {e}")
        pygame.quit()
        sys.exit(1)

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        for i, point in enumerate(calibration_positions):
            screen.fill(BACKGROUND_COLOR)
            image_rect = calib_image.get_rect(center=point)
            screen.blit(calib_image, image_rect)
            pygame.display.flip()
            logging.info(f"Displaying calibration point {i+1} at {point}")

            gaze_data = []
            start_time = time.time()
            while time.time() - start_time < calib_speed:
                try:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        logging.error("Failed to capture frame from camera")
                        break
                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb_frame)
                    if results.multi_face_landmarks:
                        gaze = get_gaze_position(frame, results.multi_face_landmarks[0])
                        if gaze:
                            gaze_data.append(gaze)
                            logging.debug(f"Gaze point collected: {gaze}")
                        else:
                            logging.debug("No face landmarks detected")
                except Exception as e:
                    logging.error(f"Error processing frame in calibration: {str(e)}")
                    break

            logging.info(f"Collected {len(gaze_data)} gaze points for point {point}")
            if gaze_data:
                logging.debug(f"Raw gaze data for point {point}: {gaze_data}")
                try:
                    gaze_data_no_outliers = remove_outliers(gaze_data)
                    logging.debug(f"Gaze data after outlier removal: {gaze_data_no_outliers}")
                    if gaze_data_no_outliers.size > 0:
                        try:
                            smoothed_gaze = smooth_gaze_data(gaze_data_no_outliers.tolist())
                            logging.debug(f"Smoothed gaze data: {smoothed_gaze}")
                            if smoothed_gaze:
                                avg_gaze = np.mean(smoothed_gaze, axis=0)
                                if not np.any(np.isnan(avg_gaze)):
                                    raw_gaze_points.append(avg_gaze)
                                    logging.info(f"Gaze data collected for point {point}: {avg_gaze}")
                                else:
                                    logging.warning(f"Invalid smoothed gaze data for point {point}")
                            else:
                                logging.warning(f"Failed to smooth gaze data for point {point}")
                        except Exception as e:
                            logging.error(f"Error processing gaze data (smoothing or averaging): {e}")
                    else:
                        logging.warning(f"No valid gaze data after outlier removal for point {point}")
                except Exception as e:
                    logging.error(f"Error in outlier removal: {e}")
            else:
                logging.warning(f"No gaze data collected for point {point}")

        logging.info(f"Collected raw gaze points: {raw_gaze_points}")
        if len(raw_gaze_points) >= 2:  # 需要至少兩個點來擬合
            raw_gaze_points_np = np.array(raw_gaze_points)
            screen_points_np = np.array(calibration_positions[:len(raw_gaze_points)]) # 確保長度一致
            try:
                logging.info("Fitting calibration model...")
                x_model = np.poly1d(np.polyfit(raw_gaze_points_np[:, 0], screen_points_np[:, 0], 3))
                y_model = np.poly1d(np.polyfit(raw_gaze_points_np[:, 1], screen_points_np[:, 1], 3))
                logging.info("Calibration complete!")
                return x_model, y_model
            except np.RankWarning as e:
                logging.error(f"Rank warning during polynomial fitting: {e}")
                logging.warning("Calibration might be inaccurate due to insufficient data points.")
                return None, None
            except np.linalg.LinAlgError as e:
                logging.error(f"Linear algebra error during polynomial fitting: {e}")
                return None, None
            except Exception as e:
                logging.error(f"Error fitting calibration model: {e}")
                return None, None
        else:
            logging.error(f"Calibration failed: Insufficient gaze points collected ({len(raw_gaze_points)}).")
            return None, None

    raw_gaze_points = np.array(raw_gaze_points)
    screen_points = np.array(calibration_positions)
    try:
        logging.info("Fitting calibration model...")
        x_model = np.poly1d(np.polyfit(raw_gaze_points[:, 0], screen_points[:, 0], 3))
        y_model = np.poly1d(np.polyfit(raw_gaze_points[:, 1], screen_points[:, 1], 3))
    except Exception as e:
        logging.error(f"Error fitting calibration model: {str(e)}")
        pygame.quit()
        sys.exit(1)

    logging.info("Calibration complete!")
    return x_model, y_model

def map_gaze_to_screen(gaze, x_model, y_model):
    try:
        mapped_x = x_model(gaze[0])
        mapped_y = y_model(gaze[1])
        return int(mapped_x), int(mapped_y)
    except Exception as e:
        logging.error(f"Error in map_gaze_to_screen: {str(e)}")
        return (0, 0)

def svop_test(screen, stimulus_positions, stimulus_diameter_pixels, calib_points, calib_speed, calib_image_path, camera_index):
    data = []
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logging.error(f"Error: Could not access webcam with index {camera_index}")
        pygame.quit()
        sys.exit(1)

    try:
        x_model, y_model = calibration_phase(screen, cap, calib_points, calib_speed, calib_image_path)
    except Exception as e:
        logging.error(f"Calibration phase failed: {e}")
        cap.release()
        pygame.quit()
        sys.exit(1)

    for stimulus in stimulus_positions:
        screen.fill(BACKGROUND_COLOR)
        pygame.draw.circle(screen, STIMULUS_COLOR, stimulus, stimulus_diameter_pixels // 2)
        pygame.display.flip()
        logging.info(f"Displaying stimulus at {stimulus}")

        gaze_data = []
        with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
            start_time = time.time()
            while time.time() - start_time < 2:
                try:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        logging.error("Failed to capture frame from camera during test")
                        break
                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb_frame)
                    if results.multi_face_landmarks:
                        gaze = get_gaze_position(frame, results.multi_face_landmarks[0])
                        if gaze:
                            mapped_gaze = map_gaze_to_screen(gaze, x_model, y_model)
                            gaze_data.append(mapped_gaze)
                except Exception as e:
                    logging.error(f"Error processing frame in test: {str(e)}")
                    break

        if len(gaze_data) > 0:
            smoothed_gaze = smooth_gaze_data(gaze_data)[-1]
            data.append({
                "stimulus_x": stimulus[0],
                "stimulus_y": stimulus[1],
                "gaze_x": smoothed_gaze[0],
                "gaze_y": smoothed_gaze[1],
                "distance": calculate_distance(stimulus, smoothed_gaze),
                "result": "PASS" if calculate_distance(stimulus, smoothed_gaze) <= 200 else "FAIL"
            })
            logging.info(f"Stimulus {stimulus} result: {data[-1]}")
        else:
            logging.warning(f"No gaze data collected for stimulus {stimulus}")

        time.sleep(1)

    cap.release()
    pygame.quit()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = f'svop_test_results_{timestamp}.csv'
    pd.DataFrame(data).to_csv(file_name, index=False)
    logging.info(f"Results saved to {file_name}!")

def calculate_distance(point1, point2):
    try:
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    except Exception as e:
        logging.error(f"Error in calculate_distance: {str(e)}")
        return float('inf')

def main_test(calib_points, calib_speed, calib_image_path, stimuli_points, camera_index):
    global SCREEN_WIDTH, SCREEN_HEIGHT, PIXELS_PER_CM
    info = pygame.display.Info()
    SCREEN_WIDTH = info.current_w
    SCREEN_HEIGHT = info.current_h
    PIXELS_PER_CM = SCREEN_WIDTH / SCREEN_WIDTH_CM
    logging.info(f"Entering main_test with screen size: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    try:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        logging.info("Screen initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize screen: {e}")
        pygame.quit()
        sys.exit(1)

    stimulus_diameter_pixels = angular_to_pixel_diameter(ANGULAR_DIAMETERS["Goldmann IV"], VIEWING_DISTANCE_CM, PIXELS_PER_CM)
    stimulus_positions = convert_positions_to_pixels(generate_points(stimuli_points), SCREEN_WIDTH, SCREEN_HEIGHT)
    svop_test(screen, stimulus_positions, stimulus_diameter_pixels, calib_points, calib_speed, calib_image_path, camera_index)

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = SVOPSettingsGUI(root)
        root.mainloop()
    except Exception as e:
        logging.error(f"Program crashed: {str(e)}")
    finally:
        logging.info("Program finished. Press Enter to exit.")
        input("Press Enter to close the console...")