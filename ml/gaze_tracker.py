"""
Gaze & Head-Pose Tracker  —  MediaPipe Tasks API (0.10.x)
Downloads face_landmarker.task (~4 MB) automatically on first run.

Smoothing: raw yaw/pitch/gaze values are averaged over the last
SMOOTH_N frames before comparing to thresholds, so brief head
movements or micro-glances no longer trigger "looking away".
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
        [  0.0,    0.0,    0.0],   # 1   nose tip
        [  0.0,  -63.6,  -12.5],  # 152 chin
        [-43.3,   32.7,  -26.0],  # 263 left eye outer
        [ 43.3,   32.7,  -26.0],  # 33  right eye outer
        [-57.5,    0.0,  -40.0],  # 234 left cheek
        [ 57.5,    0.0,  -40.0],  # 454 right cheek
    ], dtype=np.float64)
    ANCHOR_IDX = [1, 152, 263, 33, 234, 454]

    # Thresholds — raised so normal micro-movements don't fire
    YAW_DEG   = 30.0   # was 20  — needs a deliberate head turn
    PITCH_DEG = 30.0   # was 22  — needs a clear nod down/up
    GAZE_DEV  = 0.28   # was 0.22 — more iris latitude

    # Rolling-average smoothing window (frames)
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
        self._det  = mp_vision.FaceLandmarker.create_from_options(opts)
        self._ts   = 0          # monotonically-increasing timestamp (ms)

        # Smoothing buffers (store absolute deviation magnitudes)
        self._yaw_buf   = collections.deque(maxlen=self._SMOOTH_N)
        self._pitch_buf = collections.deque(maxlen=self._SMOOTH_N)
        self._gaze_buf  = collections.deque(maxlen=self._SMOOTH_N)

    # ── public ────────────────────────────────────────────────

    def process(self, frame):
        """
        Returns (analysis_dict, landmark_list_or_None, pose_tuple_or_None).
        landmark_list is a plain Python list of NormalizedLandmark objects.
        """
        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._ts += 33          # ~30 FPS — must be strictly increasing
        result = self._det.detect_for_video(mp_img, self._ts)

        if not result.face_landmarks:
            # Clear buffers so next sighting starts fresh
            self._yaw_buf.clear()
            self._pitch_buf.clear()
            self._gaze_buf.clear()
            return self._no_face(), None, None

        lms  = result.face_landmarks[0]   # list[NormalizedLandmark]
        gaze = self._iris_gaze(lms)
        yaw, pitch, pose = self._head_pose(lms, w, h)

        # Update rolling buffers with absolute deviations
        self._yaw_buf.append(abs(yaw))
        self._pitch_buf.append(abs(pitch))
        self._gaze_buf.append(abs(gaze - 0.5))

        # Only decide "away" once we have a full buffer of readings
        if len(self._yaw_buf) >= self._SMOOTH_N:
            sm_yaw   = sum(self._yaw_buf)   / len(self._yaw_buf)
            sm_pitch = sum(self._pitch_buf) / len(self._pitch_buf)
            sm_gaze  = sum(self._gaze_buf)  / len(self._gaze_buf)
        else:
            sm_yaw = sm_pitch = sm_gaze = 0.0   # warming up — stay focused

        turned   = sm_yaw   > self.YAW_DEG
        tilted   = sm_pitch > self.PITCH_DEG
        deviated = sm_gaze  > self.GAZE_DEV
        away     = turned or tilted or deviated

        reason = ('head_turned'   if turned   else
                  'head_tilted'   if tilted   else
                  'eyes_deviated' if deviated else
                  'focused')

        return {
            'face_detected': True,
            'gaze_ratio':    round(float(gaze),  3),
            'head_yaw':      round(float(yaw),   1),
            'head_pitch':    round(float(pitch), 1),
            'looking_away':  away,
            'reason':        reason,
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
                'looking_away': True, 'reason': 'no_face'}

    def _iris_gaze(self, lms):
        try:
            # Left eye : outer 263, inner 362, iris 468
            lo, li, il = lms[263].x, lms[362].x, lms[468].x
            lw = abs(li - lo)
            lg = ((il - lo) / lw) if lw > 0.003 else 0.5

            # Right eye: outer 33, inner 133, iris 473
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
