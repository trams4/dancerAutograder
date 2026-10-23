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
FPS = 30 # fps of correct.mov / data.json
VISIBILITY = 0.5
MIRROR = True # player copies the coach like a mirror (just dance style). flipping before detection makes mediapipe's left/right match the coach's
WINDOWS = [(3, 5), (6, 7), (8, 9), (10 , 11), (12, 13), (14, 15), (16, 18), (19, 20), (21, 22), (23, 24), (25, 28), (29, 30), (31, 32), (33, 34)]
NUM_ANGLES = 8
NAN_ROW = [float('nan')] * NUM_ANGLES

# setup
latest_result = None # (detection timestamp, landmarks) from the callback
latest_landmarks = []
latest_angles = []
velocities = []
live_times = [] # music time (ms) of each pose in latest_angles

with open('data.json', 'r') as file:
    data = json.load(file) # angle data from other file

# velocity = current - previous, same as the live side. frame 0 has none
data_arr = np.array(data, dtype=float)
data_vel = np.vstack([np.full((1, NUM_ANGLES), np.nan), np.diff(data_arr, axis=0)])

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
    global latest_result
    latest_result = (timestamp_ms, result.pose_landmarks)

def calculate_angle(a, b, c):
    a1 = np.array([a.x, a.y])
    b1 = np.array([b.x, b.y])
    c1 = np.array([c.x, c.y])

    ba = a1 - b1
    bc = c1 - b1

    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    ang = np.arccos(np.clip(cos, -1.0, 1.0)) # rounding can push cos past 1 and give NaN

    return np.degrees(ang)

def joint_angle(l, a, b, c):
    # hidden joints are NaN (skipped when grading) instead of 0, which looked like a wildly wrong pose
    if l[b].visibility < VISIBILITY:
        return float('nan')
    return calculate_angle(l[a], l[b], l[c])

def nan_mean(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.all(np.isnan(x)):
        return None
    return np.nanmean(x)

def music_time_ms():
    # the song is the clock, so grading stays in sync even when the loop runs under 30 fps
    if not pygame.mixer.music.get_busy():
        return -1
    return pygame.mixer.music.get_pos()

def reset_round():
    global grade_frame, flash_timer, ref_pos
    cap_correct.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ref_pos = 0
    pygame.mixer.music.play()
    grade_frame = 0
    flash_timer = 0
    latest_angles.clear()
    velocities.clear()
    live_times.clear()
    detect_music_ms.clear()

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

score = 0
user_angles = []
flash_timer = 0
current_grade = "Not Started"
grade_frame = 0
graded = False
ref_pos = 0 # next frame cap_correct will read
frame_c = None
last_detect_ts = 0 # mediapipe needs strictly increasing timestamps
handled_ts = None # detection timestamp of the last result we graded
detect_music_ms = {} # detection timestamp -> music time when that frame was captured


pygame.mixer.init()
pygame.mixer.music.load('correct.mp3')
pygame.mixer.music.play()

while cap.isOpened():
    r, frame = cap.read()
    if not r:
        break # over

    if MIRROR:
        frame = cv2.flip(frame, 1)

    now_ms = music_time_ms()
    ref_idx = int(now_ms * FPS / 1000)

    # song or reference finished -> start over
    if now_ms < 0 or ref_idx >= len(data):
        reset_round()
        continue

    # keep the reference video on the music's clock (skip frames if we fell behind)
    while ref_pos <= ref_idx:
        r_c, f = cap_correct.read()
        if not r_c:
            break
        frame_c = f
        ref_pos += 1

    # color
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # convert to mediapipe image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # send to landmarker (async, result goes to process_result callback)
    detect_ts = max(int(time.time() * 1000), last_detect_ts + 1)
    last_detect_ts = detect_ts
    detect_music_ms[detect_ts] = now_ms
    landmarker.detect_async(mp_image, detect_ts)

    # a new result arrived: record it at the music time its frame was captured, not now
    result = latest_result
    if result is not None and result[0] != handled_ts and result[0] in detect_music_ms:
        handled_ts = result[0]
        pose_ms = detect_music_ms[result[0]]
        for ts in [t for t in detect_music_ms if t <= result[0]]:
            del detect_music_ms[ts]

        latest_landmarks = result[1]
        if len(latest_landmarks) > 0:
            l = latest_landmarks[0]
            left_elbow     = joint_angle(l, 11, 13, 15)
            right_elbow    = joint_angle(l, 12, 14, 16)
            left_shoulder  = joint_angle(l, 23, 11, 13)
            right_shoulder = joint_angle(l, 24, 12, 14)
            left_knee      = joint_angle(l, 23, 25, 27)
            right_knee     = joint_angle(l, 24, 26, 28)
            left_hip       = joint_angle(l, 11, 23, 25)
            right_hip      = joint_angle(l, 12, 24, 26)

            user_angles = [left_elbow, right_elbow, left_shoulder, right_shoulder,
               left_knee, right_knee, left_hip, right_hip]
        else:
            user_angles = NAN_ROW

        # velocity = current - previous, scaled to "per reference frame" so it matches data_vel even if poses arrive slower than 30 fps
        if len(latest_angles) != 0 and pose_ms > live_times[-1]:
            frames_passed = (pose_ms - live_times[-1]) * FPS / 1000
            velocities.append(((np.array(user_angles) - np.array(latest_angles[-1])) / frames_passed).tolist())
        else:
            velocities.append(NAN_ROW)

        latest_angles.append(user_angles)
        live_times.append(pose_ms)

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

    if grade_frame < len(WINDOWS) and now_ms >= WINDOWS[grade_frame][1] * 1000:
                start_ms = WINDOWS[grade_frame][0] * 1000
                end_ms = WINDOWS[grade_frame][1] * 1000

                # every live pose in the window, paired with the reference frame at the same moment in the song
                times = np.array(live_times, dtype=float)
                in_window = np.where((times >= start_ms) & (times < end_ms))[0]
                ref_rows = np.minimum((times[in_window] * FPS / 1000).astype(int), len(data) - 1)

                live_angles = np.array(latest_angles, dtype=float).reshape(-1, NUM_ANGLES)[in_window]
                live_velocities = np.array(velocities, dtype=float).reshape(-1, NUM_ANGLES)[in_window]

                ref_angles = data_arr[ref_rows]
                ref_velocities = data_vel[ref_rows]

                # Grade section (NaN = joint hidden or no pose, left out of the average)
                diff_a = np.abs(live_angles - ref_angles)
                diff_r = np.abs(live_velocities - ref_velocities)

                curve_a = np.maximum(0, diff_a - THRESHOLD)
                curve_r = np.maximum(0, diff_r - VELOCITY_THRESHOLD)

                error_a = nan_mean(curve_a)
                error_r = nan_mean(curve_r)

                # nothing visible for the whole window = no credit
                angle_score = 0 if error_a is None else np.round(max(0, 100 - error_a))
                vel_score = 0 if error_r is None else np.round(max(0, 100 - error_r))

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
    if frame_c is not None:
        cv2.imshow("Reference", frame_c)

    flash_timer -= 1

    # key...?
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cap_correct.release()

cv2.destroyAllWindows()

pygame.quit()



