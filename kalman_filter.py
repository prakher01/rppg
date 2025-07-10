from collections import deque

hr_values = deque(maxlen=30)  

kalman_q = 0.001 
kalman_r = 2.0    


frame_counter = 0
UPDATE_HR_EVERY_N_FRAMES = 10
MAX_HR_CHANGE = 10  

def update_kalman(kalman_hr, kalman_p, bpm_smoothed):
    kalman_p += kalman_q
    kalman_k = kalman_p / (kalman_p + kalman_r)
    kalman_hr = kalman_hr + kalman_k * (bpm_smoothed - kalman_hr)
    kalman_p = (1 - kalman_k) * kalman_p
    print("kalman",kalman_hr)
    return kalman_hr, kalman_p
