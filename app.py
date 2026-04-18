"""
FocusEye — AI-Powered Procrastination Blocker
Flask + SocketIO server. Camera capture and ML run entirely in Python.
The browser receives annotated JPEG frames via WebSocket.
"""

import threading
import base64
import time
import logging

import cv2
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

from ml.focus_engine import FocusEngine
from focus.tracker   import FocusTracker
from focus.blocker   import WebsiteBlocker

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'focuseye-2024'

socketio     = SocketIO(app, cors_allowed_origins='*', async_mode='threading')
focus_engine = FocusEngine()
tracker      = FocusTracker()
blocker      = WebsiteBlocker()

_cam_running = False
_cam_thread  = None
_cam_lock    = threading.Lock()

# ─────────────────────────────────────────────────────────
# PAGE
# ─────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

# ─────────────────────────────────────────────────────────
# BLOCKED-SITES REST API
# ─────────────────────────────────────────────────────────

@app.route('/api/sites', methods=['GET'])
def get_sites():
    return jsonify(blocker.get_sites())

@app.route('/api/sites', methods=['POST'])
def add_site():
    site = request.json.get('site', '').strip().lower()
    if site:
        blocker.add_site(site)
    return jsonify({'ok': bool(site)})

@app.route('/api/sites/<path:site>', methods=['DELETE'])
def del_site(site):
    blocker.remove_site(site)
    return jsonify({'ok': True})

@app.route('/api/blocker/toggle', methods=['POST'])
def toggle_blocker():
    enable = request.json.get('enabled', False)
    result = blocker.toggle(enable)
    return jsonify(result)

@app.route('/api/blocker/status', methods=['GET'])
def blocker_status():
    return jsonify({'enabled': blocker.is_enabled})

# ─────────────────────────────────────────────────────────
# SETTINGS REST API
# ─────────────────────────────────────────────────────────

@app.route('/api/settings', methods=['GET'])
def get_settings():
    return jsonify(tracker.get_settings())

@app.route('/api/settings', methods=['POST'])
def post_settings():
    tracker.update_settings(request.json or {})
    return jsonify({'ok': True})

@app.route('/api/reset', methods=['POST'])
def reset():
    tracker.reset()
    return jsonify({'ok': True})

# ─────────────────────────────────────────────────────────
# WEBSOCKET EVENTS
# ─────────────────────────────────────────────────────────

@socketio.on('connect')
def on_connect():
    log.info(f'Client connected  sid={request.sid}')
    emit('status', {'running': _cam_running})

@socketio.on('disconnect')
def on_disconnect():
    log.info(f'Client disconnected  sid={request.sid}')

@socketio.on('start_session')
def on_start():
    global _cam_running, _cam_thread
    with _cam_lock:
        if _cam_running:
            emit('session_event', {'type': 'already_running'})
            return
        _cam_running = True
        tracker.reset()
        focus_engine.reset()        # recalibrate head/body baseline
        _cam_thread = threading.Thread(target=_camera_loop, daemon=True)
        _cam_thread.start()
    emit('session_event', {'type': 'started'})
    log.info('Session started')

@socketio.on('stop_session')
def on_stop():
    global _cam_running
    _cam_running = False
    emit('session_event', {'type': 'stopped'})
    log.info('Session stopped')

# ─────────────────────────────────────────────────────────
# CAMERA + ML LOOP  (runs in background thread)
# ─────────────────────────────────────────────────────────

def _camera_loop():
    global _cam_running

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        log.error('Cannot open camera')
        _cam_running = False
        socketio.emit('session_event', {'type': 'camera_error',
                                        'msg': 'Camera not found'})
        return

    log.info('Camera opened')
    frame_n = 0

    while _cam_running:
        ret, frame = cap.read()
        if not ret:
            log.warning('Frame read failed — retrying')
            time.sleep(0.05)
            continue

        frame   = cv2.flip(frame, 1)   # mirror selfie view
        frame_n += 1

        # ── ML analysis ──────────────────────────────
        ml      = focus_engine.analyze(frame, frame_n)
        state   = tracker.update(ml['analysis'])

        # ── Draw focus overlay onto annotated frame ──
        annotated = focus_engine.draw_overlay(ml['annotated_frame'], state)

        # ── Auto-activate blocker on alert ───────────
        if state.get('alert_needed') and blocker.is_enabled:
            blocker.activate_block()

        # ── Encode & emit ─────────────────────────────
        _, buf = cv2.imencode('.jpg', annotated,
                              [cv2.IMWRITE_JPEG_QUALITY, 72])
        img_b64 = base64.b64encode(buf).decode()

        socketio.emit('frame', {
            'image': img_b64,
            'data':  {**ml['analysis'], **state}
        })

        time.sleep(0.033)   # ~30 FPS target

    cap.release()
    _cam_running = False
    socketio.emit('session_event', {'type': 'stopped'})
    log.info('Camera released')

# ─────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 60)
    print('  FocusEye — AI-Powered Procrastination Blocker')
    print('=' * 60)
    print('  http://localhost:5000')
    print('  (Use Chrome or Edge for best results)')
    print('=' * 60)
    socketio.run(app, host='0.0.0.0', port=5000,
                 debug=False, use_reloader=False)
