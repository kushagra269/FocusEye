"""
HandPoseDetector  —  MediaPipe Tasks API (0.10.x)
Detects phone-use behaviour via two complementary signals:

  1. Hand grip  — fingers curled in a phone-holding pose
  2. Body posture — head pitched > 50° from calibrated baseline + wrist raised

Calibration: the first CALIB_READINGS pose detections record the user's
natural upright angle as baseline.  The 50° threshold is then measured as
deviation from that baseline, so sitting posture and camera angle do not matter.

Both models are downloaded automatically on first run (~8 MB + ~5 MB).
"""

import os
import math
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks.python       import vision   as mp_vision
from mediapipe.tasks.python.core  import base_options as mp_base


def _dist2d(a, b):
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

    # Deviation from calibrated baseline required to trigger phone-posture
    _POSTURE_DEG = 90.0

    # Pose readings (run_pose frames) needed to establish baseline
    _CALIB_READINGS = 15   # ~45 frames at every-3rd-frame throttle

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

        # Calibration
        self._calib_angles:  list  = []
        self._body_calibrated: bool = False
        self._baseline_angle: float = 0.0

    # ── public ────────────────────────────────────────────────

    def reset_calibration(self):
        """Call at session start to re-learn the user's natural posture."""
        self._calib_angles.clear()
        self._body_calibrated = False
        self._baseline_angle  = 0.0
        self._last = self._empty()

    @property
    def body_calibrated(self):
        return self._body_calibrated

    def process(self, frame, run_pose: bool = True) -> dict:
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # ── Hands ─────────────────────────────────────────────
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

            # Feed calibration once we have valid landmarks
            if pose_lms is not None and not self._body_calibrated:
                try:
                    angle = self._head_forward_angle(
                        pose_lms[0], pose_lms[11], pose_lms[12])
                    self._calib_angles.append(angle)
                    if len(self._calib_angles) >= self._CALIB_READINGS:
                        self._baseline_angle = (
                            sum(self._calib_angles) / len(self._calib_angles))
                        self._body_calibrated = True
                except Exception:
                    pass

            self._last['pose_lms'] = pose_lms
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
    def _head_forward_angle(nose, l_sh, r_sh) -> float:
        """
        Angle (degrees) between the shoulder-midpoint→nose vector and the
        vertical upward direction.  0 = perfectly upright, grows as the
        head pitches forward.
        """
        mid_x = (l_sh.x + r_sh.x) / 2
        mid_y = (l_sh.y + r_sh.y) / 2
        vx = nose.x - mid_x
        vy = nose.y - mid_y          # negative when nose is above shoulders
        mag = (vx ** 2 + vy ** 2) ** 0.5
        if mag < 0.001:
            return 0.0
        cos_a = max(-1.0, min(1.0, -vy / mag))
        return math.degrees(math.acos(cos_a))

    @staticmethod
    def _is_grip(lms):
        wrist  = lms[0]
        curled = 0
        for tip_i, mcp_i in [(8, 5), (12, 9), (16, 13), (20, 17)]:
            if _dist2d(lms[tip_i], wrist) < _dist2d(lms[mcp_i], wrist) * 1.15:
                curled += 1
        return curled >= 3

    @staticmethod
    def _check_wrist_raised(pose_lms):
        if pose_lms is None:
            return False
        try:
            nose       = pose_lms[0]
            shoulder_y = (pose_lms[11].y + pose_lms[12].y) / 2
            for idx in (15, 16):
                wr = pose_lms[idx]
                if wr.visibility < 0.4:
                    continue
                if (nose.y - 0.06) <= wr.y <= (shoulder_y + 0.12):
                    return True
        except Exception:
            pass
        return False

    def _check_phone_posture(self, pose_lms):
        """
        True when head pitches more than 50° forward from the calibrated
        baseline AND a wrist is raised — phone-in-hands posture.
        Stays silent until calibration is complete.
        """
        if pose_lms is None or not self._body_calibrated:
            return False
        try:
            nose  = pose_lms[0]
            l_sh  = pose_lms[11]
            r_sh  = pose_lms[12]
            angle = self._head_forward_angle(nose, l_sh, r_sh)

            # Deviation from personal baseline must exceed 50°
            deviation = angle - self._baseline_angle
            head_down = deviation > self._POSTURE_DEG

            shoulder_y = (l_sh.y + r_sh.y) / 2
            mid_y = (nose.y + shoulder_y) / 2
            wrist_up = any(
                pose_lms[idx].visibility > 0.4 and pose_lms[idx].y < mid_y + 0.05
                for idx in (15, 16)
            )
            return head_down and wrist_up
        except Exception:
            pass
        return False
