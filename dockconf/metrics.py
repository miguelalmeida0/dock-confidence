# SPDX-FileCopyrightText: 2026 Pedro Sordo Martínez <amurlaniakea@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""FR5 / SS2: calibration metrics.

Expected Calibration Error (ECE) -- standard formula (Guo et al. 2017,
"On Calibration of Modern Neural Networks"):

    ECE = Sigma_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

where n = total poses, B = number of bins (default 10, equal-width in
[0,1]), B_b = poses whose p_near_native falls in bin b,
acc(B_b) = fraction of B_b that are truly near-native (RMSD<2.0 A),
conf(B_b) = mean predicted p_near_native in B_b.

Top-1 best-pose accuracy (Eq.11, AgenticPosesRanker):
    Acc = (1/m) Sigma_i 1[ ghat_i == g_i ]
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .parse import NEAR_NATIVE_THRESHOLD, Pose


def expected_calibration_error(poses: Sequence[Pose], n_bins: int = 10) -> Optional[float]:
    """ECE per SS2.1. Returns None if no pose has p_near_native."""
    p = np.array([x.p_near_native for x in poses if x.p_near_native is not None],
                 dtype=float)
    y = np.array([1.0 if (x.near_native is True) else 0.0
                   for x in poses if x.p_near_native is not None], dtype=float)
    if p.size == 0 or p.size != y.size:
        return None

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(p, bins) - 1
    indices = np.clip(indices, 0, n_bins - 1)

    ece = 0.0
    n = p.size
    for b in range(n_bins):
        mask = indices == b
        if not mask.any():
            continue
        conf_b = float(p[mask].mean())
        acc_b = float(y[mask].mean())
        ece += (mask.sum() / n) * abs(acc_b - conf_b)
    return float(ece)


def raw_score_ece(poses: Sequence[Pose], n_bins: int = 10) -> Optional[float]:
    """ECE of the RAW (un-calibrated) score, used as the comparison
    baseline in AC7. Treats raw_score as a confidence proxy after min-max
    normalisation into [0,1] (higher score = more confident).
    """
    s = np.array([x.raw_score for x in poses if x.raw_score is not None], dtype=float)
    y = np.array([1.0 if (x.near_native is True) else 0.0
                   for x in poses if x.raw_score is not None], dtype=float)
    if s.size == 0 or s.size != y.size:
        return None
    # normalise: for affinity (more negative = better) invert.
    sources = {x.score_source for x in poses if x.raw_score is not None}
    if sources == {"smina_affinity"} or sources == {"pdbqt_score"}:
        s = -s  # lower affinity -> higher confidence
    s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.clip(np.digitize(s, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    n = s.size
    for b in range(n_bins):
        mask = indices == b
        if not mask.any():
            continue
        conf_b = float(s[mask].mean())
        acc_b = float(y[mask].mean())
        ece += (mask.sum() / n) * abs(acc_b - conf_b)
    return float(ece)


def top1_accuracy(poses: Sequence[Pose]) -> Optional[float]:
    """Top-1 best-pose accuracy (Eq.11, AgenticPosesRanker).

    Per system, pick the pose with the highest calibrated p_near_native;
    give credit if that pose is truly near-native. Return the fraction of
    systems correctly identified. Returns None if no system has a p.
    """
    by_sys: dict[str, list[Pose]] = {}
    for x in poses:
        if x.p_near_native is None:
            continue
        by_sys.setdefault(x.system_id, []).append(x)
    if not by_sys:
        return None
    correct = 0
    for sys_poses in by_sys.values():
        best = max(sys_poses, key=lambda q: q.p_near_native)  # type: ignore[arg-type]
        if best.near_native is True:
            correct += 1
    return correct / len(by_sys)


def random_baseline_accuracy(poses: Sequence[Pose]) -> float:
    """Expected random accuracy = mean_i (1 / N_i) over systems (Eq.14)."""
    systems: dict[str, int] = {}
    for x in poses:
        systems[x.system_id] = systems.get(x.system_id, 0) + 1
    if not systems:
        return 0.0
    return float(np.mean([1.0 / n for n in systems.values()]))
