# SPDX-FileCopyrightText: 2026 Pedro Sordo Martínez <amurlaniakea@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""FR5 / FR6: JSON report + reliability diagram + decoy filter."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .parse import NEAR_NATIVE_THRESHOLD, Pose


def filter_decoys(poses: Sequence[Pose], threshold: float = 0.5) -> list[Pose]:
    """Mark is_decoy=True for poses whose calibrated P(near-native) < threshold."""
    out = list(poses)
    for p in out:
        if p.p_near_native is None:
            p.is_decoy = None
        else:
            p.is_decoy = p.p_near_native < threshold
    return out


def to_dict(poses: Sequence[Pose]) -> list[dict]:
    return [
        {
            "system_id": p.system_id,
            "pose_id": p.pose_id,
            "raw_score": p.raw_score,
            "score_source": p.score_source,
            "rmsd": p.rmsd,
            "near_native": p.near_native,
            "p_near_native": p.p_near_native,
            "confidence_band": p.confidence_band,
            "is_decoy": p.is_decoy,
        }
        for p in poses
    ]


def write_json(poses: Sequence[Pose], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(to_dict(poses), fh, indent=2)


def reliability_diagram(poses: Sequence[Pose], out_path: str,
                       n_bins: int = 10) -> Optional[str]:
    """Plot conf(B_b) vs acc(B_b). Diagonal = perfect calibration.

    Returns the path if written, else None (no p_near_native data).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    p = np.array([x.p_near_native for x in poses if x.p_near_native is not None],
                   dtype=float)
    y = np.array([1.0 if x.near_native is True else 0.0
                   for x in poses if x.p_near_native is not None], dtype=float)
    if p.size == 0:
        return None

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    conf, acc = [], []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf.append(float(p[mask].mean()))
        acc.append(float(y[mask].mean()))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    ax.plot(conf, acc, "o-", label="model")
    ax.set_xlabel("Mean predicted P(near-native)")
    ax.set_ylabel("Fraction truly near-native")
    ax.set_title("Reliability Diagram (dock-confidence)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
