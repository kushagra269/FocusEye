"""
HandPoseDetector  —  MediaPipe Tasks API (0.10.x)
Detects phone-use behaviour via two complementary signals:

  1. Hand grip  — fingers curled in a phone-holding pose
  2. Body posture — head bent down + wrist raised to face/chest level

Both models are downloaded automatically on first run (~8 MB + ~5 MB).
Runs in IMAGE mode so it can be called at any throttled interval without
strict timestamp management.
"""

import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks.python       import vision   as mp_vision
from mediapipe.tasks.python.core  import base_options as mp_base


def _dist2d(a, b):
    """2-D Euclidean distance between two NormalizedLandmarks."""
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


class HandPoseDetector:

    HAND_MODEL_URL  = ('https://storage.googleapis.com/mediapipe-models/'
                       'hand_landmarker/hand_landmarker/float16/latest/'
                       'hand_landmarker.task')
    HAND_MODEL_PATH = 'hand_landmarker.task'

    POSE_MODEL_URL  = ('https://storage.googleapis.com/mediapipe-models/'
                       'pose_landmarker/pose_landmarker_lite/float16/latest/'
                       'pose_landmarker_lite.task')
    POSE_MODEL_PATH = 'pose_landmarker_lite.task'

    def __init__(self):
        self._ensure_models()

        hand_opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_base.BaseOptions(
                model_asset_path=self.HAND_MODEL_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
        self._hand_det = mp_vision.HandLandmarker.create_from_options(hand_opts)

        pose_opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_base.BaseOptions(
                model_asset_path=self.POSE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
        self._pose_det = mp_vision.PoseLandmarker.create_from_options(pose_opts)

        self._last: dict = self._empty()

    # ── public ────────────────────────────────────────────────

    def process(self, frame, run_pose: bool = True) -> dict:
        """
        Analyse `frame` for phone-use gestures.

        run_pose: pass False to skip the heavier pose model this frame
                  (focus_engine throttles it to every 3rd frame).

        Returns dict:
          hand_detected  : bool  — any hand visible
          grip_detected  : bool  — phone-grip pose on at least one hand
          wrist_raised   : bool  — wrist is at face/chest level (pose)
          posture_phone  : bool  — head-down + wrist-up combo (pose)
          hand_lms_list  : list  — list[list[NormalizedLandmark]] for drawing
          pose_lms       : list or None — 33 pose landmarks for drawing
        """
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # ── Hands (run every call) ────────────────────────────
        try:
            h_res         = self._hand_det.detect(mp_img)
            hand_lms_list = h_res.hand_landmarks
        except Exception:
            hand_lms_list = []

        hand_detected = len(hand_lms_list) > 0
        grip_detected = any(self._is_grip(lms) for lms in hand_lms_list)

        # ── Pose (throttled by caller) ────────────────────────
        if run_pose:
            try:
                p_res    = self._pose_det.detect(mp_img)
                pose_lms = p_res.pose_landmarks[0] if p_res.pose_landmarks else None
            except Exception:
                pose_lms = None
            self._last['pose_lms'] = pose_lms   # cache for skipped frames
        else:
            pose_lms = self._last.get('pose_lms')

        wrist_raised  = self._check_wrist_raised(pose_lms)
        posture_phone = self._check_phone_posture(pose_lms)

        result = {
            'hand_detected':  hand_detected,
            'grip_detected':  grip_detected,
            'wrist_raised':   wrist_raised,
            'posture_phone':  posture_phone,
            'hand_lms_list':  hand_lms_list,
            'pose_lms':       pose_lms,
        }
        self._last = result
        return result

    # ── private ───────────────────────────────────────────────

    @staticmethod
    def _empty():
        return {
            'hand_detected': False, 'grip_detected': False,
            'wrist_raised':  False, 'posture_phone': False,
            'hand_lms_list': [],    'pose_lms':      None,
        }

    def _ensure_models(self):
        for url, path, size in [
            (self.HAND_MODEL_URL, self.HAND_MODEL_PATH, '8 MB'),
            (self.POSE_MODEL_URL, self.POSE_MODEL_PATH, '5 MB'),
        ]:
            if not os.path.exists(path):
                name = os.path.basename(path)
                print(f'[HandPoseDetector] Downloading {name} (~{size})…')
                urllib.request.urlretrieve(url, path)
                print(f'[HandPoseDetector] {name} ready.')

    @staticmethod
    def _is_grip(lms):
        """
        True if the hand is in a phone-gripping pose:
        3 or more of the 4 main fingers are curled inward
        (tip not extending past its MCP knuckle relative to wrist).
        """
        wrist  = lms[0]
        curled = 0
        # (fingertip index, MCP index) for index/middle/ring/pinky
        for tip_i, mcp_i in [(8, 5), (12, 9), (16, 13), (20, 17)]:
            tip_d = _dist2d(lms[tip_i], wrist)
            mcp_d = _dist2d(lms[mcp_i], wrist)
            if tip_d < mcp_d * 1.15:   # tip not further than knuckle = curled
                curled += 1
        return curled >= 3

    @staticmethod
    def _check_wrist_raised(pose_lms):
        """
        True if either wrist is between the nose and just below the shoulders
        — the typical height for holding a phone.

        Coordinate system: y=0 at top of frame, y=1 at bottom
        (so nose.y < shoulder.y for an upright person).
        """
        if pose_lms is None:
            return False
        try:
            nose       = pose_lms[0]
            shoulder_y = (pose_lms[11].y + pose_lms[12].y) / 2
            for idx in (15, 16):   # left wrist, right wrist
                wr = pose_lms[idx]
                if wr.visibility < 0.4:
                    continue
                # wrist between slightly above nose and 12% below shoulders
                if (nose.y - 0.06) <= wr.y <= (shoulder_y + 0.12):
                    return True
        except Exception:
            pass
        return False

    @staticmethod
    def _check_phone_posture(pose_lms):
        """
        True when head is strongly pitched down AND a wrist is raised —
        the unmistakable posture of someone looking at a phone in their hands.

        "Head down" = nose y is within 10 % of the shoulder y-line
        (head has dropped forward toward the chest in camera view).
        """
        if pose_lms is None:
            return False
        try:
            nose       = pose_lms[0]
            shoulder_y = (pose_lms[11].y + pose_lms[12].y) / 2
            # nose has dropped close to shoulder level
            head_down  = nose.y > shoulder_y - 0.10
            # at least one visible wrist is above mid-point between nose and shoulder
            mid_y = (nose.y + shoulder_y) / 2
            wrist_up = any(
                pose_lms[idx].visibility > 0.4 and pose_lms[idx].y < mid_y + 0.05
                for idx in (15, 16)
            )
            return head_down and wrist_up
        except Exception:
            pass
        return False
