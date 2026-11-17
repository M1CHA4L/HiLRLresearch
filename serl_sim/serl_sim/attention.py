"""Attention-budgeted alert scheduling (no world model yet).

risk score r_t  --rank in sliding window-->  u_t in [0, 1]  --threshold tau_t-->  alert

tau_t = max(tau_B, tau_M)
  budget controller   : tau_B += eta_B * (a_t - b)            (a_t = 1 if attention used at t)
                        -> long-run attention fraction -> b  (ACI-style guarantee, any risk dist.)
  miss-rate controller: tau_M += eta_M * (alpha - err_t)      (err_t = 1 on an un-alerted failure)
                        -> spends less than b when no failures are being missed.

Risk sources implemented in train_rlpd.py: "random" (matched-budget random baseline),
"q_std" (critic-ensemble std), "neg_q" (low value = risky).
"""
from collections import deque
from typing import Optional

import numpy as np


class BudgetedAlertScheduler:
    def __init__(
        self,
        budget: float = 0.2,
        window: int = 2000,
        eta_budget: float = 0.01,
        use_miss_controller: bool = True,
        miss_alpha: float = 0.1,
        eta_miss: float = 0.05,
        miss_lookback: int = 30,  # a failure is "caught" if an alert fired within this many steps
        warmup: int = 200,
        hysteresis: int = 2,  # consecutive above-threshold steps needed to alert
        seed: Optional[int] = None,
    ):
        self.b = budget
        self.window = deque(maxlen=window)
        self.eta_b = eta_budget
        self.use_miss = use_miss_controller
        self.alpha = miss_alpha
        self.eta_m = eta_miss
        self.lookback = miss_lookback
        self.warmup = warmup
        self.hysteresis = hysteresis
        self.rng = np.random.default_rng(seed)

        self.tau_b = 1.0 - budget
        self.tau_m = 0.0
        self.t = 0
        self._above = 0
        self._last_alert_t = -10**9
        self.n_failures = 0
        self.n_missed = 0

    @property
    def tau(self) -> float:
        return max(self.tau_b, self.tau_m) if self.use_miss else self.tau_b

    def rank(self, risk: float) -> float:
        if not self.window:
            return 0.5
        w = np.fromiter(self.window, dtype=np.float64)
        return float((w < risk).mean())

    def should_alert(self, risk: float) -> bool:
        """Call once per step BEFORE env.step with the current risk score."""
        u = self.rank(risk)
        self.window.append(float(risk))
        self.t += 1
        if self.t <= self.warmup:
            alert = self.rng.random() < self.b
        else:
            self._above = self._above + 1 if u > self.tau else 0
            alert = self._above >= self.hysteresis
        if alert:
            self._last_alert_t = self.t
        self.last_rank = u
        return alert

    def observe(self, attention_used: bool, failure: bool = False):
        """Call once per step AFTER env.step."""
        a = float(attention_used)
        # NOTE: no clipping to [0, 1]. tau > 1 means "no alerts until the overspent attention is
        # paid back"; clipping would break the long-run budget guarantee
        # |mean(a) - b| <= (tau_max - tau_min + eta) / (eta * T).
        self.tau_b = float(np.clip(self.tau_b + self.eta_b * (a - self.b), -1.0, 2.0))
        if failure:
            self.n_failures += 1
            missed = (self.t - self._last_alert_t) > self.lookback
            self.n_missed += int(missed)
            if self.use_miss:
                self.tau_m = float(np.clip(self.tau_m + self.eta_m * (self.alpha - float(missed)), 0.0, 1.0))
        elif self.use_miss:
            # slow upward drift when nothing fails: spend less attention as the policy improves
            self.tau_m = float(np.clip(self.tau_m + self.eta_m * 0.01 * self.alpha, 0.0, 1.0))

    def stats(self) -> dict:
        return {
            "tau": self.tau,
            "tau_budget": self.tau_b,
            "tau_miss": self.tau_m,
            "failures": self.n_failures,
            "missed_failures": self.n_missed,
        }
