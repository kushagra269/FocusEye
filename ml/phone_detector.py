"""
Phone Detector
Uses YOLOv8-nano (ultralytics) to detect cell phones in the frame.
COCO class 67 = "cell phone".

Falls back gracefully if ultralytics is not installed.
Model weights (~6 MB) are downloaded automatically on first use.
"""

import logging

log = logging.getLogger(__name__)

# COCO class index for cell phone
_PHONE_CLASS = 67
_MIN_CONF    = 0.50


class PhoneDetector:

    def __init__(self):
        self._model    = None
        self._enabled  = False
        self._try_load()

    def _try_load(self):
        try:
            from ultralytics import YOLO          # noqa: PLC0415
            self._model   = YOLO('yolov8n.pt')    # downloads on first run
            self._enabled = True
            log.info('PhoneDetector: YOLOv8n loaded')
        except ImportError:
            log.warning(
                'ultralytics not installed — phone detection disabled.\n'
                'Run: pip install ultralytics'
            )
        except Exception as exc:
            log.warning(f'PhoneDetector init failed: {exc}')

    # ── public ────────────────────────────────────────────────

    @property
    def available(self):
        return self._enabled

    def detect(self, frame):
        """
        Returns list of dicts:
            {'bbox': [x1, y1, x2, y2], 'confidence': float}
        Returns empty list if detector unavailable or no phones found.
        """
        if not self._enabled:
            return []
        try:
            results = self._model(frame, verbose=False, conf=_MIN_CONF)
            phones  = []
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == _PHONE_CLASS:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        phones.append({
                            'bbox':       [x1, y1, x2, y2],
                            'confidence': round(float(box.conf[0]), 2),
                        })
            return phones
        except Exception as exc:
            log.debug(f'Phone detection error: {exc}')
            return []
