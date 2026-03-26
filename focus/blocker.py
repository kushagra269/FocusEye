"""
WebsiteBlocker
Adds / removes entries in the OS hosts file to block distracting sites.

Requires admin (Windows) or sudo (Linux/macOS) privileges.
Gracefully degrades if permission is denied.
"""

import os
import platform
import threading
import logging

log = logging.getLogger(__name__)

_MARKER   = '# FocusEye-block'
_REDIRECT = '127.0.0.1'

_DEFAULT_SITES = [
    'youtube.com',
    'instagram.com',
    'twitter.com',
    'x.com',
    'facebook.com',
    'tiktok.com',
    'reddit.com',
    'netflix.com',
]


class WebsiteBlocker:

    def __init__(self):
        self._lock   = threading.Lock()
        self._sites  = list(_DEFAULT_SITES)
        self.is_enabled = False
        self._hosts_path = self._detect_hosts()

    # ─────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────

    def get_sites(self) -> list:
        with self._lock:
            return list(self._sites)

    def add_site(self, site: str):
        with self._lock:
            site = site.strip().lower().removeprefix('www.')
            if site and site not in self._sites:
                self._sites.append(site)

    def remove_site(self, site: str):
        with self._lock:
            site = site.strip().lower().removeprefix('www.')
            self._sites = [s for s in self._sites if s != site]

    def toggle(self, enable: bool) -> dict:
        with self._lock:
            self.is_enabled = enable
            if enable:
                return self._write_blocks()
            return self._remove_blocks()

    def activate_block(self):
        """Called automatically when focus alert fires (if enabled)."""
        if self.is_enabled:
            with self._lock:
                self._write_blocks()

    # ─────────────────────────────────────────────────────────
    # PRIVATE
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _detect_hosts() -> str:
        if platform.system() == 'Windows':
            return r'C:\Windows\System32\drivers\etc\hosts'
        return '/etc/hosts'

    def _write_blocks(self) -> dict:
        try:
            content = self._read_hosts()
            # Remove any existing FocusEye entries
            lines = [l for l in content.splitlines() if _MARKER not in l]

            new_lines = []
            for site in self._sites:
                new_lines.append(f'{_REDIRECT} {site} {_MARKER}')
                new_lines.append(f'{_REDIRECT} www.{site} {_MARKER}')

            final = '\n'.join(lines) + '\n' + '\n'.join(new_lines) + '\n'
            self._write_hosts(final)
            log.info(f'Blocked {len(self._sites)} sites')
            return {'ok': True, 'blocked': len(self._sites)}

        except PermissionError:
            msg = ('Permission denied. '
                   'Run as Administrator (Windows) or sudo (Linux/macOS).')
            log.warning(msg)
            return {'ok': False, 'error': msg}
        except Exception as exc:
            log.error(f'Blocker write error: {exc}')
            return {'ok': False, 'error': str(exc)}

    def _remove_blocks(self) -> dict:
        try:
            content = self._read_hosts()
            lines   = [l for l in content.splitlines() if _MARKER not in l]
            self._write_hosts('\n'.join(lines) + '\n')
            log.info('Website blocking disabled')
            return {'ok': True}
        except PermissionError:
            return {'ok': False, 'error': 'Permission denied.'}
        except Exception as exc:
            return {'ok': False, 'error': str(exc)}

    def _read_hosts(self) -> str:
        with open(self._hosts_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def _write_hosts(self, content: str):
        with open(self._hosts_path, 'w', encoding='utf-8') as f:
            f.write(content)
