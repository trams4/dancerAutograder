# features.py (OUTLINE)
# shared by video.py (coach, offline) and posing.py (player, live) so both are measured the same way.
# bodies are pseudocode for now. every function says what goes in, what comes out, and the steps.
#
# pipeline per frame:
#   mediapipe pose_world_landmarks -> to_arrays -> smooth -> normalize -> 4 feature groups -> + motion
#
# the 4 groups match ubisoft's "distance, angles, orientations, motions":
#   orientations = limb directions, angles = 3d joint angles, distances = hand/foot/head positions, motions = their speeds

import numpy as np

# ---------------------------------------------------------------- constants

# mediapipe landmark ids
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# orientations: bones as (from, to). "mid_*" are computed points, not mediapipe ids
SEGMENTS = {
    "l_upper_arm": (L_SHOULDER, L_ELBOW),
    "l_forearm":   (L_ELBOW, L_WRIST),
    "r_upper_arm": (R_SHOULDER, R_ELBOW),
    "r_forearm":   (R_ELBOW, R_WRIST),
    "l_thigh":     (L_HIP, L_KNEE),
    "l_shin":      (L_KNEE, L_ANKLE),
    "r_thigh":     (R_HIP, R_KNEE),
    "r_shin":      (R_KNEE, R_ANKLE),
    "spine":       ("mid_hip", "mid_shoulder"),
    "shoulders":   (L_SHOULDER, R_SHOULDER),
    "hips":        (L_HIP, R_HIP),
    "head":        ("mid_shoulder", NOSE),
}

# angles: (a, b, c) = angle at b, same idea as calculate_angle in posing.py but in 3d
ANGLES = {
    "l_elbow":    (L_SHOULDER, L_ELBOW, L_WRIST),
    "r_elbow":    (R_SHOULDER, R_ELBOW, R_WRIST),
    "l_shoulder": (L_HIP, L_SHOULDER, L_ELBOW),
    "r_shoulder": (R_HIP, R_SHOULDER, R_ELBOW),
    "l_hip":      (L_SHOULDER, L_HIP, L_KNEE),
    "r_hip":      (R_SHOULDER, R_HIP, R_KNEE),
    "l_knee":     (L_HIP, L_KNEE, L_ANKLE),
    "r_knee":     (R_HIP, R_KNEE, R_ANKLE),
    # plus two that aren't joint triplets, computed separately in joint_angles():
    #   "torso_lean": angle between spine and straight up
    #   "torso_twist": angle between shoulder line and hip line, seen from above
}

# distances: where these points are relative to the torso
END_POINTS = {
    "l_hand": L_WRIST,
    "r_hand": R_WRIST,
    "l_foot": L_ANKLE,
    "r_foot": R_ANKLE,
    "head":   NOSE,
}

# body regions, so scoring can say "your left arm was off" and weight regions separately
REGIONS = {
    "left_arm":  ["l_upper_arm", "l_forearm", "l_elbow", "l_shoulder", "l_hand"],
    "right_arm": ["r_upper_arm", "r_forearm", "r_elbow", "r_shoulder", "r_hand"],
    "legs":      ["l_thigh", "l_shin", "r_thigh", "r_shin", "l_knee", "r_knee", "l_hip", "r_hip", "l_foot", "r_foot"],
    "torso":     ["spine", "shoulders", "hips", "head", "torso_lean", "torso_twist"],
}

VISIBILITY = 0.5   # below this a landmark counts as hidden
MOTION_SMOOTH_MS = 100  # speeds are measured over ~100 ms, not one noisy frame


# ---------------------------------------------------------------- 1. raw landmarks -> arrays

def to_arrays(world_landmarks):
    """
    in:  result.pose_world_landmarks[0]  (33 landmarks, metres, origin between the hips)
    out: joints (33, 3) float array, visibility (33,) float array

    pseudocode:
        joints = [[p.x, p.y, p.z] for p in world_landmarks]
        visibility = [p.visibility for p in world_landmarks]
        flip y so +y is up (mediapipe's y points down, like image rows)
        return np.array(joints), np.array(visibility)
    """

    joints = np.array([[p.x, p.y, p.z] for p in world_landmarks])
    visibility = np.array([p.visibility for p in world_landmarks])
    joints[:, 1] *= -1  # flips y
    return joints, visibility


# ---------------------------------------------------------------- 2. smoothing

class Filter:
    """
    jitter filter that smooths a lot when still and barely lags when moving fast.
    fixes things like data.json's right elbow jumping 66 -> 147 -> 174 degrees in 3 frames.
    one filter holds all 33 x 3 values at once.

    params (tune by eye): min_cutoff ~1.0 (lower = smoother), beta ~0.05 (higher = less lag on fast moves)

    pseudocode for __call__(x, t_ms):
        if first call: remember x, t; return x
        dt = (t_ms - last_t) / 1000
        dx = (x - last_x) / dt, low-pass it with a fixed cutoff
        cutoff = min_cutoff + beta * |dx|           # faster movement -> higher cutoff -> less smoothing
        alpha = 1 / (1 + 1 / (2 * pi * cutoff * dt))
        x_smooth = alpha * x + (1 - alpha) * last_x_smooth
        remember x_smooth, t; return x_smooth
    """

    def __init__(self, min_cutoff=1.0, beta=1, d_cutoff=1.0):
        # tuning knobs
        self.min_cutoff = min_cutoff  # Hz. smoothing when still: lower = smoother but laggier
        self.beta = beta              # how fast the cutoff rises with speed: higher = less lag on fast moves
        self.d_cutoff = d_cutoff      # Hz. smoothing on the speed estimate itself

        # memory from the previous frame (None until the first __call__)
        self.last_t = None            # ms
        self.last_x = None            # (33, 3) smoothed positions
        self.last_dx = None           # (33, 3) smoothed speed, units per second

    def __call__(self, x, t_ms):
        if self.last_t is None:
            # first frame: nothing to smooth against, so pass it through and assume nothing is moving yet
            self.last_x = x
            self.last_dx = np.zeros_like(x)
            self.last_t = t_ms
            return x

        dt = (t_ms - self.last_t) / 1000
        if dt <= 0:                      # same timestamp twice: nothing to do
            return self.last_x

        # speed, smoothed with the fixed d_cutoff
        dx = (x - self.last_x) / dt
        alpha_d = 1 / (1 + 1 / (2 * np.pi * self.d_cutoff * dt))
        dx_smooth = alpha_d * dx + (1 - alpha_d) * self.last_dx

        # position, smoothed less the faster it moves
        cutoff = self.min_cutoff + self.beta * np.abs(dx_smooth)
        alpha = 1 / (1 + 1 / (2 * np.pi * cutoff * dt))
        x_smooth = alpha * x + (1 - alpha) * self.last_x

        self.last_x = x_smooth
        self.last_dx = dx_smooth
        self.last_t = t_ms
        return x_smooth

    def reset(self):
        # call when the song restarts or the player leaves the frame
        self.last_t = None            
        self.last_x = None            # (33, 3) smoothed positions
        self.last_dx = None 
        


# ---------------------------------------------------------------- 3. normalization

def normalize(joints):
    """
    remove everything that isn't the pose itself: where you stand, how tall you are, which way you face.

    in:  joints (33, 3)
    out: joints_norm (33, 3), facing_deg (float, which way the body faces, kept as its own feature since turns are choreography)

    pseudocode:
        mid_hip = mean(joints[L_HIP], joints[R_HIP])
        mid_shoulder = mean(joints[L_SHOULDER], joints[R_SHOULDER])

        # position
        joints = joints - mid_hip

        # size: torso length = 1
        torso = norm(mid_shoulder - mid_hip)
        joints = joints / torso

        # facing: rotate around the vertical (y) axis so the hip line points along +x
        hip_vec = joints[R_HIP] - joints[L_HIP], ignore its y part
        facing_deg = atan2(hip_vec.z, hip_vec.x) in degrees
        joints = rotate_y(joints, -facing_deg)

        return joints, facing_deg
    """
    raise NotImplementedError


def extra_points(joints):
    """
    the computed points SEGMENTS refers to.
    out: {"mid_hip": (3,), "mid_shoulder": (3,)}
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 4. the four feature groups

def limb_directions(joints):
    """
    ORIENTATIONS. which way each bone points, independent of bone length.
    this is the big upgrade: a straight arm pointing up and one pointing forward have the same elbow
    angle, but different directions.

    out: {segment_name: unit vector (3,)}

    pseudocode:
        for name, (a, b) in SEGMENTS:
            v = point(b) - point(a)        # point() handles "mid_*" names via extra_points
            dirs[name] = v / norm(v)
    """
    raise NotImplementedError


def joint_angles(joints):
    """
    ANGLES. 3d version of calculate_angle, in degrees.

    out: {angle_name: degrees}

    pseudocode:
        for name, (a, b, c) in ANGLES:
            ba, bc = joints[a] - joints[b], joints[c] - joints[b]
            angles[name] = degrees(arccos(clip(dot(ba, bc) / (norm(ba) * norm(bc)), -1, 1)))
        angles["torso_lean"]  = angle between spine direction and (0, 1, 0)
        angles["torso_twist"] = angle between shoulder line and hip line, both flattened onto the floor (y = 0)
    """
    raise NotImplementedError


def end_positions(joints):
    """
    DISTANCES. where hands, feet and head are, in torso lengths from the hip centre.
    catches "right shape, wrong place" (arm bent correctly but held too low).

    out: {point_name: (3,)}

    pseudocode:
        return {name: joints[idx] for name, idx in END_POINTS}
    """
    raise NotImplementedError


def confidences(visibility):
    """
    how much to trust each feature, from mediapipe visibility. a feature is only as visible as its least
    visible landmark. used as weights when comparing (replaces "set hidden joints to 0").

    out: {feature_name: 0..1}  (same keys as the three dicts above)

    pseudocode:
        for each segment / angle / end point:
            conf = min(visibility of the landmarks it uses)
            conf = 0 if conf < VISIBILITY else conf
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 5. motion

def motion(history, t_ms):
    """
    MOTIONS. how fast and which way things are moving, measured over ~MOTION_SMOOTH_MS so it isn't noise.

    in:  history = recent (t_ms, frame_features) pairs, newest last
    out: {
        "dir_vel":   {segment: (3,)  change in direction per second},
        "angle_vel": {angle: degrees per second},
        "end_vel":   {point: (3,)  torso lengths per second},
    }

    pseudocode:
        old = the entry in history closest to t_ms - MOTION_SMOOTH_MS
        dt = (t_ms - old.t) / 1000
        if dt <= 0: return all NaN
        each velocity = (current value - old value) / dt
    note: per SECOND, not per frame, so 20 fps live and 30 fps reference line up without any fixing.
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 6. putting it together

class FeatureExtractor:
    """
    one per dancer (one for the coach in video.py, one for the player in posing.py).
    keeps the smoothing filter and recent history between frames.

    usage:
        fx = FeatureExtractor()
        feats = fx.update(result.pose_world_landmarks, t_ms)   # every frame with a result
        feats = fx.update(None, t_ms)                          # frame with no pose -> all NaN, conf 0
        fx.reset()                                             # song restarts
    """

    def __init__(self):
        # self.filter = OneEuroFilter()
        # self.history = deque(maxlen=~30)
        raise NotImplementedError

    def update(self, world_landmarks, t_ms):
        """
        out: dict with keys "t_ms", "facing_deg", "dirs", "angles", "ends", "conf", "motion"

        pseudocode:
            if world_landmarks is None: return empty_features(t_ms)
            joints, vis = to_arrays(world_landmarks)
            joints = self.filter(joints, t_ms)
            joints, facing = normalize(joints)
            feats = {dirs, angles, ends from the functions above, conf from confidences(vis), facing}
            self.history.append((t_ms, feats))
            feats["motion"] = motion(self.history, t_ms)
            return feats
        """
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


def empty_features(t_ms):
    """same shape as a normal frame, every value NaN, every confidence 0."""
    raise NotImplementedError


def to_vector(feats):
    """
    flatten one frame's features into a fixed-order 1d array (+ matching confidence array).
    this is what DTW and the scoring model in scoring.py consume.
    order: dirs (12 x 3), angles (10), ends (5 x 3), facing (1), then the same for motion.

    out: values (N,), weights (N,)
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 7. comparing one player frame to one coach frame

def frame_distance(player, coach):
    """
    per-region, per-group distance between two frames. scoring.py runs this inside DTW and
    averages it over each window.

    out: {(region, group): distance}  e.g. ("left_arm", "dirs"): 0.12

    pseudocode, per feature, weighted by min(player conf, coach conf) and skipping NaN:
        dirs:   1 - dot(p, c)                        0 = same direction, 2 = opposite
        angles: |p - c| / 180                        0..1
        ends:   norm(p - c)                          torso lengths
        motion: speed: |norm(p) - norm(c)| / (norm(c) + small)
                direction: 1 - cosine(p, c), only where the coach is actually moving
        then average inside each (region, group) using REGIONS
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 8. saving the coach

def save_reference(path, frames):
    """
    video.py calls this instead of writing data.json.
    frames = list of FeatureExtractor outputs, one per video frame (NaN frames included so timing stays exact).

    pseudocode:
        times = [f["t_ms"] for f in frames]
        vectors, weights = zip(*[to_vector(f) for f in frames])
        np.savez(path, t_ms=times, values=vectors, weights=weights, version=1)
    """
    raise NotImplementedError


def load_reference(path):
    """
    out: t_ms (F,), values (F, N), weights (F, N)
    posing.py looks up the coach at song time t with: row = searchsorted(t_ms, t)
    """
    raise NotImplementedError
