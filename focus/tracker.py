"""
FocusTracker
Maintains all focus-session state based on per-frame ML analysis.

Frame-smoothing prevents micro-glances from triggering a distraction event.
Calculates focus score as percentage of focused time in the session.
Generates alert signals when away-time or event-count thresholds are hit.
"""

import time
import threading


class FocusTracker:

    # Smoothing: frames of consecutive "away" needed to start an event
    _FRAMES_TO_START = 20   # ~0.67 s at 30 FPS
    _FRAMES_TO_END   = 15   # ~0.5  s

    def __init__(self):
        self._lock = threading.Lock()
        self._settings = {
            'away_time_limit_s':   120,   # 2 minutes
            'phone_alert_instant': True,
            'sensitivity':         'medium',   # low | medium | high
        }
        self._state = {}
        self._reset_state()

    # ─────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────

    def update(self, analysis: dict) -> dict:
        """
        Called every frame with the ML analysis dict.
        Returns the current focus state dict (merged into the SocketIO payload).
        """
        with self._lock:
            return self._update(analysis)

    def reset(self):
        with self._lock:
            self._reset_state()

    def get_settings(self) -> dict:
        with self._lock:
            return dict(self._settings)

    def update_settings(self, data: dict):
        with self._lock:
            allowed = {'away_time_limit_s', 'phone_alert_instant', 'sensitivity'}
            for k, v in data.items():
                if k in allowed:
                    self._settings[k] = v
            self._apply_sensitivity()

    # ─────────────────────────────────────────────────────────
    # PRIVATE
    # ─────────────────────────────────────────────────────────

    def _reset_state(self):
        now = time.time() * 1000
        self._state = {
            # smoothing counters
            'away_frames':  0,
            'focus_frames': 0,
            # distraction episode
            'is_distracted':         False,
            'distraction_start_ms':  None,
            # accumulated stats
            'total_away_ms':   0.0,
            'away_events':     0,
            'phone_events':    0,
            'phone_start_ms':  None,
            # session
            'session_start_ms': now,
            # alert state
            'alert_needed':  False,
            'alert_ack_time': 0.0,
            'alert_cooldown_ms': 20_000,
        }
        self._apply_sensitivity()

    def _apply_sensitivity(self):
        s = self._settings.get('sensitivity', 'medium')
        if s == 'low':
            self._state['away_frames_threshold'] = 30
        elif s == 'high':
            self._state['away_frames_threshold'] = 10
        else:   # medium
            self._state['away_frames_threshold'] = 20

    def _update(self, analysis: dict) -> dict:
        now  = time.time() * 1000
        st   = self._state
        lim  = self._settings['away_time_limit_s'] * 1000   # ms

        currently_away = (analysis.get('looking_away', False)
                          or analysis.get('phone_detected', False))

        # ── Frame counters ─────────────────────────────────
        thresh = st.get('away_frames_threshold', self._FRAMES_TO_START)
        if currently_away:
            st['away_frames']  += 1
            st['focus_frames']  = 0
        else:
            st['focus_frames'] += 1
            st['away_frames']   = 0

        # ── Start distraction episode ─────────────────────
        if (not st['is_distracted']
                and st['away_frames'] >= thresh):
            st['is_distracted']        = True
            st['distraction_start_ms'] = now
            st['away_events']         += 1

        # ── End distraction episode ───────────────────────
        if (st['is_distracted']
                and st['focus_frames'] >= self._FRAMES_TO_END):
            if st['distraction_start_ms']:
                st['total_away_ms'] += now - st['distraction_start_ms']
                st['distraction_start_ms'] = None
            st['is_distracted'] = False

        # ── Live away time (current episode) ─────────────
        live_ms = ((now - st['distraction_start_ms'])
                   if st['is_distracted'] and st['distraction_start_ms']
                   else 0.0)
        total_away = st['total_away_ms'] + live_ms

        # ── Phone event counting ──────────────────────────
        phone_now = analysis.get('phone_detected', False)
        if phone_now and st['phone_start_ms'] is None:
            st['phone_start_ms'] = now
        if not phone_now:
            if st['phone_start_ms'] and (now - st['phone_start_ms']) > 2000:
                st['phone_events'] += 1
            st['phone_start_ms'] = None

        # ── Focus score ───────────────────────────────────
        session_ms = now - st['session_start_ms']
        focus_score = (max(0, round(100 * (1 - total_away / session_ms)))
                       if session_ms > 2000 else 100)

        # ── Alert logic ──────────────────────────────────
        time_exceeded  = total_away >= lim
        event_exceeded = st['away_events'] >= 30
        phone_instant  = (self._settings['phone_alert_instant']
                          and phone_now
                          and st['phone_start_ms']
                          and (now - st['phone_start_ms']) > 3000)

        cooldown_ok = (now - st['alert_ack_time']) > st['alert_cooldown_ms']
        alert_needed = (time_exceeded or event_exceeded or phone_instant) and cooldown_ok

        if alert_needed:
            st['alert_ack_time'] = now   # start cooldown immediately

        return {
            'is_distracted':   st['is_distracted'],
            'total_away_ms':   round(total_away),
            'live_away_ms':    round(live_ms),
            'away_events':     st['away_events'],
            'phone_events':    st['phone_events'],
            'focus_score':     focus_score,
            'session_ms':      round(session_ms),
            'alert_needed':    alert_needed,
            'phone_detected':  phone_now,
            'away_limit_ms':   lim,
        }
