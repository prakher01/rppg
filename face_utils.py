import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

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
