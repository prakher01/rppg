from flask import Blueprint, request, Response, send_from_directory
import os
import uuid
import tempfile
import cv2
import json
from face_utils import create_face_mask_with_colors, skin_segmentation
from signal_utils import blackout_outside_dynamic_threshold, extract_rgb_signals, chrom_method, rgb_buffer, compute_heart_rate, kalman_hr

video_blueprint = Blueprint('video_routes', __name__)
TEMP_DIR = tempfile.gettempdir()

@video_blueprint.route('/upload', methods=['POST'])
def upload():
    video_file = request.files['video']
    temp_filename = f"{uuid.uuid4()}.mp4"
    temp_path = os.path.join(TEMP_DIR, temp_filename)
    video_file.save(temp_path)
    return Response(process_video(temp_path), mimetype='text/event-stream')

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_filename = os.path.join('static', 'frame.jpg')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        face_mask = create_face_mask_with_colors(frame)

        if face_mask is None or not face_mask.any():
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
            if heart_rate == 75.0:
                heart_rate = 'Detecting'
                cv2.putText(frame, "HR: Detecting",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                hr_text = f"{heart_rate:.2f} BPM"
                cv2.putText(frame, f"HR: {hr_text}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imwrite(frame_filename, frame)
        data = {
            'frame_url': '/static/frame.jpg',
            'rgb': rgb_values,
            'heart_rate': heart_rate
        }
        yield f"data: {json.dumps(data)}\n\n"
        cv2.waitKey(30)

    cap.release()
    if os.path.exists(video_path):
        os.remove(video_path)

@video_blueprint.route('/static/<filename>')
def serve_static(filename):
    return send_from_directory('static', filename)
