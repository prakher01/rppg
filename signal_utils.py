import numpy as np
from scipy.signal import butter, filtfilt
from collections import deque
from kalman_filter import update_kalman
import cv2



BUFFER_SIZE = 300
FPS = 30
rgb_buffer = {'r': deque(maxlen=BUFFER_SIZE), 'g': deque(maxlen=BUFFER_SIZE), 'b': deque(maxlen=BUFFER_SIZE)}
hr_values = deque(maxlen=10)
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

def extract_rgb_signals(frame):
    face_pixels = frame[frame.sum(axis=2) > 0]
    if len(face_pixels) == 0:
        return (0, 0, 0)
    mean_r = int(np.mean(face_pixels[:, 2]))
    mean_g = int(np.mean(face_pixels[:, 1]))
    mean_b = int(np.mean(face_pixels[:, 0]))
    return (mean_r, mean_g, mean_b)

def chrom_method(r_signal, g_signal, b_signal):
    X = 3 * np.array(r_signal) - 2 * np.array(g_signal)
    Y = 1.5 * np.array(g_signal) - 1.5 * np.array(b_signal)
    return X + Y

def bandpass_filter(signal, lowcut=0.6, highcut=3.0, fs=FPS, order=5):
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal) if len(signal) > order else signal

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
    kalman_hr, kalman_p = update_kalman(kalman_hr, kalman_p, bpm_smoothed)

    return kalman_hr
