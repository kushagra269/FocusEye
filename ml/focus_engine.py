"""
FocusEngine
Orchestrates GazeTracker + PhoneDetector + HandPoseDetector,
draws all OpenCV annotations onto the frame, and returns the
annotated frame plus a flat analysis dict.

phone_detected is True when ANY of these fire:
  • YOLOv8 sees a physical phone in frame
  • HandPoseDetector sees a grip gesture + wrist at phone height
  • HandPoseDetector sees head-down + wrist-raised body posture

All drawing is done with raw OpenCV — no mediapipe.solutions dependency.

Color convention (BGR):
  Purple  #7850ff → (255,  80, 120)
  Teal    #00d4aa → (170, 212,   0)
  Amber   #ffaa30 → ( 48, 170, 255)
  Red     #ff3c3c → ( 60,  60, 255)
"""

import cv2
import numpy as np

from .gaze_tracker      import GazeTracker
from .phone_detector    import PhoneDetector
from .hand_pose_detector import HandPoseDetector

# BGR color palette
C_PURPLE = (255,  80, 120)
C_TEAL   = (170, 212,   0)
C_AMBER  = ( 48, 170, 255)
C_RED    = ( 60,  60, 255)
C_WHITE  = (230, 226, 255)
C_DIM    = ( 60,  40,  90)
C_GREEN  = ( 60, 200,  60)

# Throttle intervals (frames)
_PHONE_INTERVAL    = 5   # YOLO phone detector
_POSE_INTERVAL     = 3   # body pose (heavier model)

# ── Face mesh connection chains ────────────────────────────────────────────
_FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
]
_LEFT_EYE  = [33, 7, 163, 144, 145, 153, 154, 155, 133,
              173, 157, 158, 159, 160, 161, 246, 33]
_RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263,
              466, 388, 387, 386, 385, 384, 398, 362]
_LEFT_BROW  = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
_RIGHT_BROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
_LIPS_OUT  = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
              291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61]
_LIPS_IN   = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
              308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]
_NOSE_BRIDGE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]
_NOSE_TIP    = [279, 278, 344, 440, 275, 4, 45, 220, 115, 49]

_LEFT_IRIS  = [469, 470, 471, 472, 469]
_RIGHT_IRIS = [474, 475, 476, 477, 474]

_CONTOUR_CHAINS = [
    _FACE_OVAL, _LEFT_EYE, _RIGHT_EYE,
    _LEFT_BROW, _RIGHT_BROW,
    _LIPS_OUT, _LIPS_IN,
    _NOSE_BRIDGE, _NOSE_TIP,
]

# ── Hand skeleton connections (21 landmarks) ─────────────────────────────
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # thumb
    (0,5),(5,6),(6,7),(7,8),         # index
    (0,9),(9,10),(10,11),(11,12),    # middle
    (0,13),(13,14),(14,15),(15,16),  # ring
    (0,17),(17,18),(18,19),(19,20),  # pinky
    (5,9),(9,13),(13,17),            # palm
]

# ── Pose skeleton connections (upper body only) ────────────────────────
_POSE_CONNECTIONS = [
    (11, 12),           # shoulders
    (11, 13),(13, 15),  # left arm
    (12, 14),(14, 16),  # right arm
    (11, 23),(12, 24),  # torso
    (23, 24),           # hips
]


class FocusEngine:

    def __init__(self):
        self.gaze      = GazeTracker()
        self.phones    = PhoneDetector()
        self.hand_pose = HandPoseDetector()

        self._last_phones: list = []
        self._last_hp: dict    = {
            'hand_detected': False, 'grip_detected': False,
            'wrist_raised':  False, 'posture_phone': False,
            'hand_lms_list': [],    'pose_lms':      None,
        }

    def reset(self):
        """Re-run calibration at the start of a new session."""
        self.gaze.reset_calibration()
        self.hand_pose.reset_calibration()
        self._last_phones = []
        self._last_hp = {
            'hand_detected': False, 'grip_detected': False,
            'wrist_raised':  False, 'posture_phone': False,
            'hand_lms_list': [],    'pose_lms':      None,
        }

    # ─────────────────────────────────────────────────────────
    # PUBLIC: analyze
    # ─────────────────────────────────────────────────────────

    def analyze(self, frame: np.ndarray, frame_n: int = 0) -> dict:
        """
        Run all ML on `frame`.
        Returns {'annotated_frame': np.ndarray, 'analysis': dict}.
        """
        h, w  = frame.shape[:2]
        ann   = frame.copy()

        # ── Gaze + head pose ──────────────────────────────────
        gaze_data, face_lms, pose = self.gaze.process(frame)

        # ── YOLO phone detection (throttled) ─────────────────
        if frame_n % _PHONE_INTERVAL == 0:
            self._last_phones = self.phones.detect(frame)
        phone_list = self._last_phones

        # ── Hand + body-pose detection (throttled) ────────────
        run_pose = (frame_n % _POSE_INTERVAL == 0)
        self._last_hp = self.hand_pose.process(frame, run_pose=run_pose)
        hp = self._last_hp

        # ── Combined phone signal ─────────────────────────────
        phone_yolo    = len(phone_list) > 0
        phone_gesture = hp['grip_detected'] and hp['wrist_raised']
        phone_posture = hp['posture_phone']
        phone_detected = phone_yolo or phone_gesture or phone_posture

        phone_reason = ('yolo'    if phone_yolo    else
                        'gesture' if phone_gesture else
                        'posture' if phone_posture else None)

        # ── Draw face mesh ────────────────────────────────────
        if face_lms:
            self._draw_mesh(ann, face_lms, w, h)
            if pose:
                self._draw_pose_axes(ann, *pose, w, h)
            self._draw_gaze_arrow(ann, face_lms, gaze_data, w, h)

        # ── Draw hand skeletons ───────────────────────────────
        for hand_lms in hp['hand_lms_list']:
            grip = hp['grip_detected']
            self._draw_hand(ann, hand_lms, w, h, grip)

        # ── Draw pose skeleton ────────────────────────────────
        if hp['pose_lms']:
            self._draw_body_pose(ann, hp['pose_lms'], w, h,
                                 phone_posture or phone_gesture)

        # ── Draw YOLO phone boxes ─────────────────────────────
        for p in phone_list:
            self._draw_phone(ann, p)

        # ── Phone-gesture banner (no YOLO box needed) ─────────
        if (phone_gesture or phone_posture) and not phone_yolo:
            self._draw_gesture_banner(ann, w, phone_reason)

        # ── Calibration progress bar (shown while warming up) ──
        if not self.gaze.calibrated:
            self._draw_calibration_bar(ann, self.gaze.calib_progress, w, h)

        analysis = {
            'face_detected':  gaze_data['face_detected'],
            'gaze_ratio':     gaze_data['gaze_ratio'],
            'head_yaw':       gaze_data['head_yaw'],
            'head_pitch':     gaze_data['head_pitch'],
            'looking_away':   gaze_data['looking_away'],
            'reason':         gaze_data['reason'],
            'screen_facing':  gaze_data.get('screen_facing', False),
            'phone_detected': phone_detected,
            'phone_count':    len(phone_list),
            'phone_reason':   phone_reason,
            'hand_detected':  hp['hand_detected'],
            'grip_detected':  hp['grip_detected'],
            'wrist_raised':   hp['wrist_raised'],
            'posture_phone':  phone_posture,
        }
        return {'annotated_frame': ann, 'analysis': analysis}

    # ─────────────────────────────────────────────────────────
    # PUBLIC: draw_overlay
    # ─────────────────────────────────────────────────────────

    def draw_overlay(self, frame: np.ndarray, state: dict) -> np.ndarray:
        """Draw focus-state overlays (red border, score, status text)."""
        h, w = frame.shape[:2]
        out  = frame.copy()

        is_distr      = state.get('is_distracted', False)
        focus_score   = state.get('focus_score',   100)
        total_away_ms = state.get('total_away_ms',   0)
        away_events   = state.get('away_events',      0)
        phone_now     = state.get('phone_detected',  False)

        # ── Red border when distracted ─────────────────────
        if is_distr:
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), C_RED, 7)

        # ── Phone banner ───────────────────────────────────
        if phone_now:
            banner = '  SMARTPHONE DETECTED  '
            (bw, bh), _ = cv2.getTextSize(
                banner, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            bx = (w - bw) // 2
            cv2.rectangle(out, (bx - 8, 8), (bx + bw + 8, 8 + bh + 14),
                          (0, 40, 180), -1)
            cv2.putText(out, banner, (bx, 8 + bh + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_AMBER, 2)

        # ── Bottom-left status ─────────────────────────────
        label = (f'DISTRACTED  {self._fmt(total_away_ms)}'
                 if is_distr else 'FOCUSED')
        col   = C_RED if is_distr else C_TEAL
        cv2.putText(out, label, (12, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        # ── Top-right: focus score ─────────────────────────
        score_col = (C_TEAL  if focus_score > 70 else
                     C_AMBER if focus_score > 40 else C_RED)
        cv2.putText(out, f'FOCUS  {focus_score:3d}%',
                    (w - 175, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, score_col, 2)

        # ── Top-right second line: events ──────────────────
        cv2.putText(out, f'EVENTS {away_events:3d}',
                    (w - 175, 58), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, C_DIM, 1)

        return out

    # ─────────────────────────────────────────────────────────
    # PRIVATE: drawing helpers
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _lm_px(lms, idx, w, h):
        lm = lms[idx]
        return (int(lm.x * w), int(lm.y * h))

    def _draw_chain(self, frame, lms, chain, color, thickness, w, h):
        pts = [self._lm_px(lms, i, w, h) for i in chain if i < len(lms)]
        for a, b in zip(pts, pts[1:]):
            cv2.line(frame, a, b, color, thickness)

    def _draw_mesh(self, frame, lms, w, h):
        n = len(lms)
        for i in range(min(468, n)):
            cv2.circle(frame, self._lm_px(lms, i, w, h), 1, C_DIM, -1)
        for chain in _CONTOUR_CHAINS:
            valid = [i for i in chain if i < n]
            if len(valid) >= 2:
                self._draw_chain(frame, lms, valid, C_PURPLE, 1, w, h)
        if n >= 478:
            self._draw_chain(frame, lms, _LEFT_IRIS,  C_TEAL, 2, w, h)
            self._draw_chain(frame, lms, _RIGHT_IRIS, C_TEAL, 2, w, h)
            cv2.circle(frame, self._lm_px(lms, 468, w, h), 3, C_TEAL, -1)
            cv2.circle(frame, self._lm_px(lms, 473, w, h), 3, C_TEAL, -1)

    def _draw_pose_axes(self, frame, rvec, tvec, nose_2d, cam, dist, w, h):
        try:
            import numpy as np
            axis_len = int(min(w, h) * 0.07)
            pts, _ = cv2.projectPoints(
                np.float32([[axis_len, 0, 0],
                            [0, axis_len, 0],
                            [0, 0, axis_len]]),
                rvec, tvec, cam, dist)
            p0 = nose_2d
            cv2.arrowedLine(frame, p0, tuple(map(int, pts[0].ravel())),
                            (0, 0, 220), 2, tipLength=0.25)
            cv2.arrowedLine(frame, p0, tuple(map(int, pts[1].ravel())),
                            (0, 200, 0), 2, tipLength=0.25)
            cv2.arrowedLine(frame, p0, tuple(map(int, pts[2].ravel())),
                            (220, 100, 0), 2, tipLength=0.25)
        except Exception:
            pass

    def _draw_gaze_arrow(self, frame, lms, gaze_data, w, h):
        try:
            nx = int(lms[1].x * w)
            ny = int(lms[1].y * h)
            iris_offset  = (gaze_data['gaze_ratio'] - 0.5) * 160
            pitch_offset = gaze_data['head_pitch'] * 1.8
            ex = int(nx + iris_offset)
            ey = int(ny + pitch_offset)
            col = C_TEAL if abs(iris_offset) < 24 else C_RED
            cv2.arrowedLine(frame, (nx, ny), (ex, ey), col, 2, tipLength=0.35)
        except Exception:
            pass

    def _draw_hand(self, frame, hand_lms, w, h, grip: bool):
        """Draw hand skeleton; amber if grip detected, teal otherwise."""
        col = C_AMBER if grip else C_TEAL
        for a_i, b_i in _HAND_CONNECTIONS:
            if a_i >= len(hand_lms) or b_i >= len(hand_lms):
                continue
            a = (int(hand_lms[a_i].x * w), int(hand_lms[a_i].y * h))
            b = (int(hand_lms[b_i].x * w), int(hand_lms[b_i].y * h))
            cv2.line(frame, a, b, col, 2)
        # Landmark dots
        for lm in hand_lms:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, col, -1)

    def _draw_body_pose(self, frame, pose_lms, w, h, alert: bool):
        """Draw upper-body pose skeleton; red if phone-use posture detected."""
        col = C_RED if alert else C_GREEN
        for a_i, b_i in _POSE_CONNECTIONS:
            if a_i >= len(pose_lms) or b_i >= len(pose_lms):
                continue
            a_lm, b_lm = pose_lms[a_i], pose_lms[b_i]
            if a_lm.visibility < 0.4 or b_lm.visibility < 0.4:
                continue
            a = (int(a_lm.x * w), int(a_lm.y * h))
            b = (int(b_lm.x * w), int(b_lm.y * h))
            cv2.line(frame, a, b, col, 2)
        # Joint dots for key upper-body points
        for idx in (0, 11, 12, 13, 14, 15, 16):
            if idx >= len(pose_lms):
                continue
            lm = pose_lms[idx]
            if lm.visibility < 0.4:
                continue
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, col, -1)

    def _draw_phone(self, frame, phone: dict):
        x1, y1, x2, y2 = phone['bbox']
        conf = phone['confidence']
        cv2.rectangle(frame, (x1, y1), (x2, y2), C_AMBER, 2)
        cv2.putText(frame, f'PHONE  {conf:.0%}',
                    (x1, max(y1 - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, C_AMBER, 2)

    @staticmethod
    def _draw_gesture_banner(frame, w, reason):
        """Small banner for gesture/posture-based phone detection."""
        labels = {
            'gesture': 'PHONE GRIP DETECTED',
            'posture': 'PHONE POSTURE DETECTED',
        }
        text = labels.get(reason, 'PHONE USE DETECTED')
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        bx = (w - tw) // 2
        by = 45
        cv2.rectangle(frame, (bx - 8, by - th - 6),
                      (bx + tw + 8, by + 6), (0, 40, 150), -1)
        cv2.putText(frame, text, (bx, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_AMBER, 2)

    @staticmethod
    def _draw_calibration_bar(frame, progress: float, w, h):
        """
        Show a progress bar at the bottom of the frame during calibration.
        Progress 0.0 – 1.0.  Disappears once calibration is complete.
        """
        text = 'CALIBRATING — look at screen naturally'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = h - 40
        cv2.putText(frame, text, ((w - tw) // 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_AMBER, 1)
        # Bar background
        bx, by = 20, h - 22
        bar_w  = w - 40
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 10), C_DIM, -1)
        # Filled portion
        filled = int(bar_w * progress)
        if filled > 0:
            cv2.rectangle(frame, (bx, by), (bx + filled, by + 10), C_TEAL, -1)

    @staticmethod
    def _fmt(ms: float) -> str:
        s = int(ms / 1000)
        return f'{s // 60}m {s % 60:02d}s' if s >= 60 else f'{s}s'
