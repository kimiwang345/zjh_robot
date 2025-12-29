# config/config_loader.py
import json
import os
import time

class HotConfig:
    def __init__(self, path: str, reload_interval: float = 1.0):
        self.path = path
        self.reload_interval = reload_interval
        self._last_load = 0
        self._mtime = 0
        self.data = {}

        self.load(force=True)

    def load(self, force=False):
        now = time.time()
        if not force and now - self._last_load < self.reload_interval:
            return

        self._last_load = now

        try:
            mtime = os.path.getmtime(self.path)
            if mtime != self._mtime:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                self._mtime = mtime
                print(f"[HotConfig] reloaded: {self.path}", flush=True)
        except Exception as e:
            print(f"[HotConfig] load failed: {e}", flush=True)

    def get(self):
        self.load()
        return self.data
