import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import json

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),       # face
    (0,4),(4,5),(5,6),(6,8),       # face other side
    (11,12),                        # shoulders
    (11,13),(13,15),               # left arm
    (12,14),(14,16),               # right arm
    (11,23),(12,24),(23,24),       # torso
    (23,25),(25,27),               # left leg
    (24,26),(26,28),               # right leg
]

def calculate_angle(a, b, c):
    a1 = np.array([a.x, a.y])
    b1 = np.array([b.x, b.y])
    c1 = np.array([c.x, c.y])

    ba = a1 - b1
    bc = c1 - b1

    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    ang = np.arccos(np.clip(cos, -1.0, 1.0)) # rounding can push cos past 1 and give NaN

    return np.degrees(ang)


base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

landmarker = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture("correct.mov")


angs = []

while cap.isOpened():
    r, frame = cap.read()

    if not r:
        break # over

    # color
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # convert to mediapipe image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # send to landmarker (async, result goes to process_result callback)
    result = landmarker.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

    if len(result.pose_landmarks) == 0:
        angs.append([float('nan')] * 8) # keep one row per frame so data.json stays lined up with the video

    if len(result.pose_landmarks) > 0:
        h, w, _ =  frame.shape

        for l in result.pose_landmarks:
            # Grade
            left_elbow    = calculate_angle(l[11], l[13], l[15])
            right_elbow   = calculate_angle(l[12], l[14], l[16])
            left_shoulder = calculate_angle(l[23], l[11], l[13])
            right_shoulder= calculate_angle(l[24], l[12], l[14])
            left_knee     = calculate_angle(l[23], l[25], l[27])
            right_knee    = calculate_angle(l[24], l[26], l[28])
            left_hip      = calculate_angle(l[11], l[23], l[25])
            right_hip     = calculate_angle(l[12], l[24], l[26])

            user_angles = [left_elbow, right_elbow, left_shoulder, right_shoulder, 
               left_knee, right_knee, left_hip, right_hip]
            
            angs.append(user_angles)

            for p in l:
                ex = p.x * w
                why = p.y * h
                cv2.circle(frame, (int(ex), int(why)), 5,  (0, 255, 0), 3)
            for c in CONNECTIONS:
                p1 = l[c[0]] 
                p2 = l[c[1]]
                cv2.line(frame, (int(p1.x * w), int(p1.y * h)),  (int(p2.x * w), int(p2.y * h)), (0, 255, 0), 2)
        
    cv2.imshow("Pose Estimation", frame)
    

    # key...?
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


with open("data.json", "w") as f:
    json.dump(angs, f)
            



