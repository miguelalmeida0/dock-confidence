# SPDX-FileCopyrightText: 2026 Pedro Sordo Martínez <amurlaniakea@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""FR4: calibrate raw dock score -> P(RMSD < 2.0 A).

Two modes:
  - 'platt'     : 1-D logistic regression on (raw_score, near_native)
                   fit on a calibration set (CASF-2016 / PDBbind train).
  - 'heuristic' : binned piecewise mapping when no calibration set is
                   available. Documented as LESS precise (AC4 still holds,
                   but ECE will be higher -- reported honestly, never faked).

DiffDock emits a raw 'confidence' that the authors themselves warn is
hard to compare across complexes and is NOT a direct binding measure
(DiffDock README SS 'How to interpret the output confidence score').
Our job is precisely to CALIBRATE that score against RMSD<2.0 A.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .parse import NEAR_NATIVE_THRESHOLD, Pose


# ---------------------------------------------------------------------------
# Platt scaling (logistic) -- no sklearn, plain numpy
# ---------------------------------------------------------------------------
def _sigmoid(z: float) -> float:
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def fit_platt(scores: Sequence[float], labels: Sequence[int]):
    """Fit logistic P(y=1) = sigmoid(a*s + b) via Newton-Raphson."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    # standardise for stability
    mu, sigma = float(s.mean()), float(s.std()) + 1e-9
    z = (s - mu) / sigma
    a, b = 0.0, 0.0
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-(a * z + b)))
        grad_a = np.sum((p - y) * z)
        grad_b = np.sum(p - y)
        # Hessian (diagonal approx, 2nd deriv of cross-entropy)
        h_aa = np.sum(p * (1 - p) * z * z) + 1e-6
        h_bb = np.sum(p * (1 - p)) + 1e-6
        step_a = grad_a / h_aa
        step_b = grad_b / h_bb
        a -= step_a
        b -= step_b
        if abs(step_a) < 1e-6 and abs(step_b) < 1e-6:
            break
    return float(a / sigma), float(b - a * mu / sigma)


def platt_predict(coef: tuple[float, float], score: float) -> float:
    a, b = coef
    return _sigmoid(a * float(score) + b)


# ---------------------------------------------------------------------------
# Heuristic piecewise (fallback, no calibration set)
# ---------------------------------------------------------------------------
def _heuristic_map(score: float, source: Optional[str]) -> float:
    """Tramos documentados. DiffDock confidence: c>0 high, -1.5<c<0 mod,
    c<-1.5 low (README). Smina affinity: more negative = better -> invert.
    """
    if source == "diffdock_confidence":
        if score > 0.0:
            return 0.85
        if score > -1.5:
            return 0.55
        return 0.20
    if source in ("smina_affinity", "pdbqt_score"):
        # affinity: more NEGATIVE = better pose. Map -14..-4 -> 0.9..0.1
        s = max(-14.0, min(-4.0, float(score)))
        return float(0.9 - (s - (-14.0)) / 10.0 * 0.8)
    # generic fallback
    s = max(-1.0, min(1.0, float(score)))
    return float((s + 1.0) / 2.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def calibrate(poses: Sequence[Pose],
              mode: str = "heuristic",
              train: Optional[Sequence[Pose]] = None) -> list[Pose]:
    """Attach p_near_native + confidence_band to each Pose.

    mode='platt' requires `train` (poses with raw_score + near_native).
    mode='heuristic' uses documented tramos.
    Returns the same list with fields populated (mutates in place).
    """
    if mode == "platt":
        if train is None:
            raise ValueError("platt mode requires a calibration `train` set")
        tr = [t for t in train if t.raw_score is not None and t.near_native is not None]
        if not tr:
            raise ValueError("train set has no (raw_score, near_native) pairs")
        coef = fit_platt([t.raw_score for t in tr],
                          [1 if t.near_native else 0 for t in tr])
        for p in poses:
            if p.raw_score is None:
                p.p_near_native = None
                p.confidence_band = None
                continue
            prob = platt_predict(coef, p.raw_score)
            p.p_near_native = float(prob)
            p.confidence_band = _band(prob)
    else:  # heuristic
        for p in poses:
            if p.raw_score is None:
                p.p_near_native = None
                p.confidence_band = None
                continue
            prob = _heuristic_map(p.raw_score, p.score_source)
            p.p_near_native = float(prob)
            p.confidence_band = _band(prob)
    return list(poses)


def _band(p: float) -> str:
    if p >= 0.66:
        return "high"
    if p >= 0.33:
        return "moderate"
    return "low"
