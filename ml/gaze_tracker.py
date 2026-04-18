"""
Gaze & Head-Pose Tracker  —  MediaPipe Tasks API (0.10.x)
Downloads face_landmarker.task (~4 MB) automatically on first run.

Calibration: the first CALIB_FRAMES frames where a face is detected are used
to record the user's natural "looking at screen" position as baseline (0°).
All thresholds are then measured as deviation from that baseline, so sitting
position and camera angle do not matter.

Smoothing: after calibration, values are averaged over SMOOTH_N frames so
brief head movements do not trigger distraction.
"""

import os
import collections
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python       import vision   as mp_vision
from mediapipe.tasks.python.core  import base_options as mp_base


class GazeTracker:
    # Model auto-download
    MODEL_URL  = ('https://storage.googleapis.com/mediapipe-models/'
                  'face_landmarker/face_landmarker/float16/latest/'
                  'face_landmarker.task')
    MODEL_PATH = 'face_landmarker.task'

    # 3-D face model (6 anchor points, mm)
    FACE_3D = np.array([
        [  0.0,    0.0,    0.0],
        [  0.0,  -63.6,  -12.5],
        [-43.3,   32.7,  -26.0],
        [ 43.3,   32.7,  -26.0],
        [-57.5,    0.0,  -40.0],
        [ 57.5,    0.0,  -40.0],
    ], dtype=np.float64)
    ANCHOR_IDX = [1, 152, 263, 33, 234, 454]

    # Deviation-from-baseline thresholds (distraction detection).
    # Calibration captures the user's natural working position over 15 s.
    # If the head moves beyond these amounts FROM that calibrated baseline,
    # the user is considered distracted.
    YAW_DEG   = 20.0   # degrees of horizontal turn allowed from calibrated position
    PITCH_DEG = 15.0   # degrees of vertical nod allowed from calibrated position
    GAZE_DEV  = 0.20   # iris ratio deviation allowed from calibrated center

    # Lenient absolute thresholds for "screen_facing" re-focus check.
    # These are NOT relative to baseline — just "is the user generally
    # looking toward the screen?" so they can recover from any position.
    REFOCUS_YAW_ABS   = 45.0   # absolute yaw  (°) — head within ±45° of camera
    REFOCUS_PITCH_ABS = 35.0   # absolute pitch (°) — head within ±35° of camera

    # Calibration: how many face-detected frames to average for the baseline
    _CALIB_FRAMES = 450   # 15 s at 30 FPS — gives a stable personal baseline

    # Smoothing window after calibration
    _SMOOTH_N = 6

    def __init__(self):
        self._ensure_model()
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_base.BaseOptions(
                model_asset_path=self.MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._det = mp_vision.FaceLandmarker.create_from_options(opts)
        self._ts  = 0

        self._yaw_buf   = collections.deque(maxlen=self._SMOOTH_N)
        self._pitch_buf = collections.deque(maxlen=self._SMOOTH_N)
        self._gaze_buf  = collections.deque(maxlen=self._SMOOTH_N)

        # Calibration state
        self._calib_yaw   = []
        self._calib_pitch = []
        self._calib_gaze  = []
        self._calibrated  = False
        self._baseline_yaw   = 0.0
        self._baseline_pitch = 0.0
        self._baseline_gaze  = 0.5

    # ── public ────────────────────────────────────────────────

    def reset_calibration(self):
        """Call this at the start of each session to re-learn the baseline."""
        self._calib_yaw.clear()
        self._calib_pitch.clear()
        self._calib_gaze.clear()
        self._calibrated  = False
        self._baseline_yaw   = 0.0
        self._baseline_pitch = 0.0
        self._baseline_gaze  = 0.5
        self._yaw_buf.clear()
        self._pitch_buf.clear()
        self._gaze_buf.clear()

    @property
    def calibrated(self):
        return self._calibrated

    @property
    def calib_progress(self):
        """0.0 – 1.0 progress through the calibration phase."""
        return min(1.0, len(self._calib_yaw) / self._CALIB_FRAMES)

    def process(self, frame):
        """
        Returns (analysis_dict, landmark_list_or_None, pose_tuple_or_None).
        During calibration looking_away is always False and reason='calibrating'.
        """
        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._ts += 33
        result = self._det.detect_for_video(mp_img, self._ts)

        if not result.face_landmarks:
            self._yaw_buf.clear()
            self._pitch_buf.clear()
            self._gaze_buf.clear()
            return self._no_face(), None, None

        lms  = result.face_landmarks[0]
        gaze = self._iris_gaze(lms)
        yaw, pitch, pose = self._head_pose(lms, w, h)

        # ── Calibration phase ─────────────────────────────────
        if not self._calibrated:
            self._calib_yaw.append(yaw)
            self._calib_pitch.append(pitch)
            self._calib_gaze.append(gaze)
            if len(self._calib_yaw) >= self._CALIB_FRAMES:
                n = len(self._calib_yaw)
                self._baseline_yaw   = sum(self._calib_yaw)   / n
                self._baseline_pitch = sum(self._calib_pitch) / n
                self._baseline_gaze  = sum(self._calib_gaze)  / n
                self._calibrated = True
            # Stay focused during calibration — don't penalise user
            return {
                'face_detected': True,
                'gaze_ratio':    round(float(gaze),  3),
                'head_yaw':      round(float(yaw),   1),
                'head_pitch':    round(float(pitch), 1),
                'looking_away':  False,
                'reason':        'calibrating',
                'calibrated':    False,
                'screen_facing': True,   # always facing screen during calibration
            }, lms, pose

        # ── Post-calibration: deviation from personal baseline ─
        self._yaw_buf.append(abs(yaw   - self._baseline_yaw))
        self._pitch_buf.append(abs(pitch - self._baseline_pitch))
        self._gaze_buf.append(abs(gaze  - self._baseline_gaze))

        if len(self._yaw_buf) >= self._SMOOTH_N:
            sm_yaw   = sum(self._yaw_buf)   / len(self._yaw_buf)
            sm_pitch = sum(self._pitch_buf) / len(self._pitch_buf)
            sm_gaze  = sum(self._gaze_buf)  / len(self._gaze_buf)
        else:
            sm_yaw = sm_pitch = sm_gaze = 0.0

        turned   = sm_yaw   > self.YAW_DEG
        tilted   = sm_pitch > self.PITCH_DEG
        deviated = sm_gaze  > self.GAZE_DEV
        away     = turned or tilted or deviated

        reason = ('head_turned'   if turned   else
                  'head_tilted'   if tilted   else
                  'eyes_deviated' if deviated else
                  'focused')

        # Lenient "facing screen" check — used for re-focus recovery.
        # Uses raw absolute head angles, NOT deviation from personal baseline,
        # so the user can recover from ANY comfortable screen-facing position.
        screen_facing = (abs(yaw)   <= self.REFOCUS_YAW_ABS and
                         abs(pitch) <= self.REFOCUS_PITCH_ABS)

        return {
            'face_detected': True,
            'gaze_ratio':    round(float(gaze),  3),
            'head_yaw':      round(float(yaw),   1),
            'head_pitch':    round(float(pitch), 1),
            'looking_away':  away,
            'reason':        reason,
            'calibrated':    True,
            'screen_facing': screen_facing,
        }, lms, pose

    # ── private ───────────────────────────────────────────────

    def _ensure_model(self):
        if not os.path.exists(self.MODEL_PATH):
            print(f'[GazeTracker] Downloading face model to {self.MODEL_PATH} (~4 MB)…')
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            print('[GazeTracker] Model ready.')

    @staticmethod
    def _no_face():
        return {'face_detected': False, 'gaze_ratio': 0.5,
                'head_yaw': 0.0, 'head_pitch': 0.0,
                'looking_away': True, 'reason': 'no_face',
                'calibrated': False, 'screen_facing': False}

    def _iris_gaze(self, lms):
        try:
            lo, li, il = lms[263].x, lms[362].x, lms[468].x
            lw = abs(li - lo)
            lg = ((il - lo) / lw) if lw > 0.003 else 0.5

            ro, ri, ir = lms[33].x, lms[133].x, lms[473].x
            rw = abs(ro - ri)
            rg = ((ir - ri) / (ro - ri)) if rw > 0.003 else 0.5

            return float(np.clip((lg + rg) / 2, 0, 1))
        except Exception:
            return 0.5

    def _head_pose(self, lms, w, h):
        try:
            face_2d = np.array(
                [[lms[i].x * w, lms[i].y * h] for i in self.ANCHOR_IDX],
                dtype=np.float64)
            f   = float(w)
            cam = np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], dtype=np.float64)
            dist= np.zeros((4,1), dtype=np.float64)

            ok, rvec, tvec = cv2.solvePnP(
                self.FACE_3D, face_2d, cam, dist,
                flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return 0.0, 0.0, None

            rmat, _ = cv2.Rodrigues(rvec)
            ang, *_ = cv2.RQDecomp3x3(rmat)
            yaw   = float(ang[1]) * 360.0
            pitch = float(ang[0]) * 360.0
            nose  = tuple(map(int, face_2d[0]))
            return yaw, pitch, (rvec, tvec, nose, cam, dist)
        except Exception:
            return 0.0, 0.0, None
