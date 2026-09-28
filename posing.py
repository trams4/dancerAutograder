import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np
import json
import pygame

ANGLE_WEIGHT = 0.7
VELOCITY_WEIGHT = 0.3
THRESHOLD = 20
VELOCITY_THRESHOLD = 0
FPS = 30
WINDOWS = [(3, 5), (6, 7), (8, 9), (10 , 11), (12, 13), (14, 15), (16, 18), (19, 20), (21, 22), (23, 24), (25, 28), (29, 30), (31, 32), (33, 34)]

# setup
latest_landmarks = []
data_vel = []
data_vel.append([0,0,0,0,0,0,0,0]) # for frame 0 and so angles and vels match sizewise
latest_angles = []
velocities = []

with open('data.json', 'r') as file:
    data = json.load(file) # angle data from other file 

for i in range(len(data) - 1):
    data_vel.append((np.array(data[i + 1]) - np.array(data[i])).tolist())

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

# funcs
def process_result(result, output_image, timestamp_ms):
    global latest_landmarks
    latest_landmarks = result.pose_landmarks

def calculate_angle(a, b, c):
    a1 = np.array([a.x, a.y])
    b1 = np.array([b.x, b.y])
    c1 = np.array([c.x, c.y])

    ba = a1 - b1
    bc = c1 - b1

    ang = np.arccos( np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)))

    return np.degrees(ang)

base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=process_result
)

landmarker = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap_correct = cv2.VideoCapture("correct.mov")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

counter = 0
score = 0
user_angles = []
flash_timer = 0
current_grade = "Not Started"
grade_frame = 0
graded = False


pygame.mixer.init()
pygame.mixer.music.load('correct.mp3')
pygame.mixer.music.play()

while cap.isOpened():
    r, frame = cap.read()
    frame = cv2.flip(frame, 1)
    r_c, frame_c = cap_correct.read()

    
    if not r:
        break # over
    
    if not r_c:
        cap_correct.set(cv2.CAP_PROP_POS_FRAMES, 0)
        counter = 0
        pygame.mixer.music.play()
        grade_frame = 0
        flash_timer = 0
        latest_angles.clear()
        velocities.clear()
        continue
    

    # color
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # convert to mediapipe image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # send to landmarker (async, result goes to process_result callback)
    landmarker.detect_async(mp_image, int(time.time() * 1000))

    if len(latest_landmarks) > 0:
        h, w, _ =  frame.shape

        for l in latest_landmarks:
            
            for p in l:
                ex = p.x * w
                why = p.y * h
                cv2.circle(frame, (int(ex), int(why)), 5,  (0, 255, 0), 3)
            for c in CONNECTIONS:
                p1 = l[c[0]] 
                p2 = l[c[1]]
                cv2.line(frame, (int(p1.x * w), int(p1.y * h)),  (int(p2.x * w), int(p2.y * h)), (0, 255, 0), 2)
            
            # Grade
            left_elbow     = calculate_angle(l[11], l[13], l[15]) if l[13].visibility > 0.5 else 0
            right_elbow    = calculate_angle(l[12], l[14], l[16]) if l[14].visibility > 0.5 else 0
            left_shoulder  = calculate_angle(l[23], l[11], l[13]) if l[11].visibility > 0.5 else 0
            right_shoulder = calculate_angle(l[24], l[12], l[14]) if l[12].visibility > 0.5 else 0
            left_knee      = calculate_angle(l[23], l[25], l[27]) if l[25].visibility > 0.5 else 0
            right_knee     = calculate_angle(l[24], l[26], l[28]) if l[26].visibility > 0.5 else 0
            left_hip       = calculate_angle(l[11], l[23], l[25]) if l[23].visibility > 0.5 else 0
            right_hip      = calculate_angle(l[12], l[24], l[26]) if l[24].visibility > 0.5 else 0
            

            user_angles = [left_elbow, right_elbow, left_shoulder, right_shoulder, 
               left_knee, right_knee, left_hip, right_hip]  

            if len(latest_angles) != 0:
                velocities.append((np.array(latest_angles[-1]) - np.array(user_angles)).tolist())
            else:
                velocities.append([0,0,0,0,0,0,0,0])

            latest_angles.append(user_angles)
    else:
        latest_angles.append([0,0,0,0,0,0,0,0])
        velocities.append([0,0,0,0,0,0,0,0])

    if grade_frame < len(WINDOWS) and counter + 1 == WINDOWS[grade_frame][1] * FPS:
                # Snip the two lists by WINDOWS[grade_frame] 
                live_angles = latest_angles[(WINDOWS[grade_frame][0] * FPS):((WINDOWS[grade_frame][1] * FPS) + 1)]
                live_velocities = velocities[(WINDOWS[grade_frame][0] * FPS):((WINDOWS[grade_frame][1] * FPS) + 1)]

                ref_angles = data[(WINDOWS[grade_frame][0] * FPS):((WINDOWS[grade_frame][1] * FPS))]
                ref_velocities = data_vel[(WINDOWS[grade_frame][0] * FPS):((WINDOWS[grade_frame][1] * FPS))]

                # Grade section
                diff_a = np.abs(np.array(live_angles) - np.array(ref_angles))
                diff_r = np.abs(np.array(live_velocities) - np.array(ref_velocities))

                curve_a = np.maximum(0, diff_a - THRESHOLD)
                curve_r = np.maximum(0, diff_r - VELOCITY_THRESHOLD)

                error_a = np.mean(curve_a)
                error_r = np.mean(curve_r)

                angle_score = np.round(max(0, 100 - error_a))
                vel_score = np.round(max(0, 100 - error_r))

                score = angle_score * ANGLE_WEIGHT + vel_score * VELOCITY_WEIGHT
                # ok
                if score > 100:
                    score = 100
                if score >= 90 and score <= 100:
                    current_grade = "Perfect"
                elif score >= 80 and score < 90:
                    current_grade = "Super"
                elif score >= 70 and score < 80:
                    current_grade = "Good"
                elif score >= 60 and score < 70:
                    current_grade = "Ok"
                else:
                    current_grade = "X"
                
                graded = True
                grade_frame += 1
    # show!
    if graded or flash_timer > 0:
        if graded:
            flash_timer = 90
            graded = False
        
        cv2.putText(

            frame,

            f"Score: {int(score)}",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 255, 255),

            2

        )

        cv2.putText(

            frame,

            current_grade,

            (20, 80),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 255, 0),

            2

        )
    

    cv2.imshow("Pose Estimation", frame)
    cv2.imshow("Reference", frame_c)
    
    counter += 1
    flash_timer -= 1

    # key...?
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cap_correct.release()

cv2.destroyAllWindows()

pygame.quit()



