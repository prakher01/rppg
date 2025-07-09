import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt
from collections import deque
import os
import json

# MediaPipe FaceMesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Buffers for signal
BUFFER_SIZE = 300
FPS = 30
rgb_buffer = {'r': deque(maxlen=BUFFER_SIZE), 'g': deque(maxlen=BUFFER_SIZE), 'b': deque(maxlen=BUFFER_SIZE)}
hr_values = deque(maxlen=10)

# Kalman filter for HR smoothing
kalman_hr = 75.0
kalman_p = 1.0
kalman_q = 0.01
kalman_r = 1.0

def blackout_outside_dynamic_threshold(frame, lower_factor=0.48, upper_factor=1.74):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    non_zero_pixels = gray_image[gray_image != 0]
    average_value = np.median(non_zero_pixels)
    lower_threshold = max(0, int(average_value * lower_factor))
    upper_threshold = min(255, int(average_value * upper_factor))
    mask = (gray_image >= lower_threshold) & (gray_image <= upper_threshold)
    updated_frame = np.zeros_like(frame)
    updated_frame[mask] = frame[mask]
    return updated_frame

def bandpass_filter(signal, lowcut=0.6, highcut=3.0, fs=FPS, order=5):
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal) if len(signal) > order else signal

def chrom_method(r_signal, g_signal, b_signal):
    X = 3 * np.array(r_signal) - 2 * np.array(g_signal)
    Y = 1.5 * np.array(g_signal) - 1.5 * np.array(b_signal)
    return X + Y

def compute_heart_rate(signal):
    global kalman_hr, kalman_p

    if len(signal) < BUFFER_SIZE:
        return kalman_hr

    filtered_signal = bandpass_filter(signal)
    fft_spectrum = np.fft.rfft(filtered_signal)
    freqs = np.fft.rfftfreq(len(filtered_signal), d=1/FPS)

    valid_range = (freqs >= 0.92) & (freqs <= 2.0)
    if not any(valid_range):
        return kalman_hr

    peak_freq = freqs[valid_range][np.argmax(np.abs(fft_spectrum[valid_range]))]
    bpm = peak_freq * 60
    if bpm < 40:
        bpm = kalman_hr

    hr_values.append(bpm)
    bpm_smoothed = np.mean(hr_values)

    kalman_p += kalman_q
    kalman_k = kalman_p / (kalman_p + kalman_r)
    kalman_hr = kalman_hr + kalman_k * (bpm_smoothed - kalman_hr)
    kalman_p = (1 - kalman_k) * kalman_p

    return kalman_hr

def extract_rgb_signals(frame):
    face_pixels = frame[frame.sum(axis=2) > 0]
    if len(face_pixels) == 0:
        return (0, 0, 0)
    mean_r = int(np.mean(face_pixels[:, 2]))
    mean_g = int(np.mean(face_pixels[:, 1]))
    mean_b = int(np.mean(face_pixels[:, 0]))
    return (mean_r, mean_g, mean_b)

def create_face_mask_with_colors(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            points = np.array([(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
                               for landmark in face_landmarks.landmark], dtype=np.int32)
            hull = cv2.convexHull(points)
            cv2.fillPoly(mask, [hull], 255)

    masked_face = cv2.bitwise_and(image, image, mask=mask)
    return masked_face

def skin_segmentation(face_mask):
    hsv = cv2.cvtColor(face_mask, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    return cv2.bitwise_and(face_mask, face_mask, mask=skin_mask)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_filename = os.path.join('static', 'frame.jpg')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        face_mask = create_face_mask_with_colors(frame)

        if face_mask is None or not np.any(face_mask):
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            rgb_values = {'r': 0, 'g': 0, 'b': 0}
            heart_rate = 0.0
        else:
            skin_segmented = skin_segmentation(face_mask)
            face = blackout_outside_dynamic_threshold(skin_segmented)
            mean_r, mean_g, mean_b = extract_rgb_signals(face)
            rgb_buffer["r"].append(mean_r)
            rgb_buffer["g"].append(mean_g)
            rgb_buffer["b"].append(mean_b)
            pulse_signal = chrom_method(rgb_buffer['r'], rgb_buffer['g'], rgb_buffer['b'])
            heart_rate = compute_heart_rate(pulse_signal)

            rgb_values = {'r': mean_r, 'g': mean_g, 'b': mean_b}
            hr_text = f"{heart_rate:.2f} BPM" if kalman_hr != 75.0 else "Detecting"
            cv2.putText(frame, f"R: {mean_r}, G: {mean_g}, B: {mean_b}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"HR: {hr_text}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imwrite(frame_filename, frame)
        data = {
            'frame_url': '/static/frame.jpg',
            'rgb': rgb_values,
            'heart_rate': heart_rate if kalman_hr != 75.0 else None
        }
        yield f"data: {json.dumps(data)}\n\n"
        cv2.waitKey(30)

    cap.release()
    if os.path.exists(video_path):
        os.remove(video_path)