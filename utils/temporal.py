"""Monotonic, frame-rate-independent gesture commits. Nothing edits text here."""

from dataclasses import dataclass
from math import isfinite

from utils.recognition import Candidate, Decision

_MAX_GAP = 0.25
_MAX_INTERRUPTION = 0.15
_RELEASE = 0.30
_EPSILON = 1e-9  # Keep inclusive decimal boundaries stable in binary floats.


@dataclass(frozen=True, slots=True)
class Timing:
    alphabet_hold: float = 0.45
    custom_hold: float = 0.70
    destructive_hold: float = 1.0
    cooldown: float = 0.35
    destructive_cooldown: float = 1.25

    def __post_init__(self):
        values = self.alphabet_hold, self.custom_hold, self.destructive_hold, self.cooldown, self.destructive_cooldown
        if not all(isfinite(value) and value > 0 for value in values):
            raise ValueError("Hold and cooldown durations must be finite and positive")
        if self.destructive_hold < max(self.alphabet_hold, self.custom_hold):
            raise ValueError("Destructive hold cannot be shorter than alphabet or custom holds")
        if self.destructive_cooldown < self.cooldown:
            raise ValueError("Destructive cooldown cannot be shorter than global cooldown")


class Stabilizer:
    def __init__(self, timing=Timing(), *, arm_destructive_on_start: bool = False):
        self.timing = timing
        self._arm_on_start = bool(arm_destructive_on_start)
        self.reset()

    def reset(self):
        self._last_seen = None
        self._release_since = None
        self._release_observations = 0
        self._last_label = None
        self._last_commit = float("-inf")
        self._last_destructive = float("-inf")
        self.destructive_armed = self._arm_on_start
        self._waiting_release = False
        self._forget_candidate()

    @property
    def waiting_release(self) -> bool:
        return self._waiting_release

    @property
    def progress(self) -> float:
        if self._candidate is None:
            return 0.0
        if self._candidate.destructive:
            hold = self.timing.destructive_hold
        elif self._candidate.source == "alphabet":
            hold = self.timing.alphabet_hold
        else:
            hold = self.timing.custom_hold
        return min(1.0, max(0.0, self._elapsed / hold))

    def _forget_candidate(self):
        self._candidate = None
        self._elapsed = 0.0
        self._accepted = 0
        self._interrupted = None

    def observe(self, decision: Decision, now: float) -> Candidate | None:
        previous = self._last_seen
        if not isfinite(now) or (previous is not None and now <= previous):
            return None  # Late or duplicate timestamps cannot advance progress or rearm.
        self._last_seen = now
        self._waiting_release = False
        if previous is None or now - previous > _MAX_GAP + _EPSILON:
            self._forget_candidate()
            self._release_since = None
            self._release_observations = 0

        if decision.kind == "no-hands":
            if self._release_since is None:
                self._release_since = now
            self._release_observations += 1
            if self._release_observations >= 2 and now - self._release_since + _EPSILON >= _RELEASE:
                self._last_label = None
                self.destructive_armed = True
        else:
            self._release_since = None
            self._release_observations = 0

        if decision.kind == "ambiguous":
            self.destructive_armed = False
            self._forget_candidate()
            return None

        current = decision.candidate if decision.kind == "usable" else None
        if current is None:
            if self._candidate is not None:
                if self._interrupted is None:
                    self._interrupted = previous
                if self._candidate.destructive or now - self._interrupted > _MAX_INTERRUPTION + _EPSILON:
                    self._forget_candidate()
            return None

        if current.label == self._last_label:
            self._waiting_release = True
            self._forget_candidate()
            return None
        if current.destructive and not self.destructive_armed:
            self._forget_candidate()
            return None

        if current != self._candidate or (
            self._interrupted is not None and now - self._interrupted > _MAX_INTERRUPTION + _EPSILON
        ):
            self._forget_candidate()
            self._candidate = current
        elif self._interrupted is None:
            self._elapsed += now - previous
        self._interrupted = None
        self._accepted += 1

        if current.destructive:
            hold = self.timing.destructive_hold
            if now - self._last_destructive + _EPSILON < self.timing.destructive_cooldown:
                return None
        else:
            hold = self.timing.alphabet_hold if current.source == "alphabet" else self.timing.custom_hold
        if (self._accepted < 3 or self._elapsed + _EPSILON < hold
                or now - self._last_commit + _EPSILON < self.timing.cooldown):
            return None

        self._last_commit = now
        self._last_label = current.label
        if current.destructive:
            self.destructive_armed = False
            self._last_destructive = now
        self._forget_candidate()
        return current
