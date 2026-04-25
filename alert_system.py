"""
alert_system.py
---------------
Manages real-time alert generation with cooldown deduplication.
Each event_type has its OWN independent cooldown, so Violence + Gun
can both fire simultaneously for the same frame.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AlertRecord:
    event_type: str
    video_timestamp: str
    confidence: float
    wall_time: float = field(default_factory=time.time)
    message: str = ""

    def __post_init__(self):
        emoji_map = {
            "violence": "🚨",
            "weapon":   "🔫",
            "gun":      "🔫",
            "knife":    "🔪",
            "fire":     "🔥",
            "smoke":    "💨",
        }
        emoji = "⚠️"
        for key, em in emoji_map.items():
            if key in self.event_type.lower():
                emoji = em
                break
        self.message = (
            f"{emoji} ALERT: {self.event_type} detected "
            f"at {self.video_timestamp} "
            f"[conf: {self.confidence:.2f}]"
        )

    def to_dict(self) -> dict:
        return {
            "event_type":      self.event_type,
            "video_timestamp": self.video_timestamp,
            "confidence":      round(self.confidence, 4),
            "message":         self.message,
        }


class AlertSystem:
    """
    Alert manager where EVERY event_type has its own independent cooldown.
    This means Violence + Gun + Fire can all raise alerts simultaneously
    for the same frame — they never block each other.
    """

    SEVERITY_ERROR   = {"violence", "gun", "weapon", "fire"}
    SEVERITY_WARNING = {"knife", "smoke"}

    def __init__(self, cooldown_seconds: float = 2.0, max_history: int = 200):
        self._cooldown = cooldown_seconds
        # Each event_type gets its own last-fired timestamp
        self._last_alert: dict = {}
        self.history: deque = deque(maxlen=max_history)

    def _on_cooldown(self, event_type: str) -> bool:
        """Check cooldown PER event_type independently."""
        last = self._last_alert.get(event_type.lower(), 0.0)
        return (time.time() - last) < self._cooldown

    def check_and_raise(
        self,
        event_type: str,
        video_timestamp: str,
        confidence: float,
    ) -> Optional[AlertRecord]:
        """
        Raise an alert for event_type if not on cooldown.
        Each event_type (Violence, Gun, Knife, Fire, Smoke) is tracked
        INDEPENDENTLY — a Violence alert never suppresses a Gun alert.
        """
        key = event_type.lower()
        if self._on_cooldown(key):
            return None

        record = AlertRecord(event_type, video_timestamp, confidence)
        self._last_alert[key] = record.wall_time
        self.history.append(record)
        return record

    def severity(self, event_type: str) -> str:
        for key in self.SEVERITY_ERROR:
            if key in event_type.lower():
                return "error"
        return "warning"

    def recent_alerts(self, n: int = 10) -> list:
        """Return N most recent alerts as dicts, newest first."""
        return [r.to_dict() for r in reversed(list(self.history))][:n]

    def clear(self):
        self.history.clear()
        self._last_alert.clear()