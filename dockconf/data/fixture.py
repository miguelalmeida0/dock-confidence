"""T12/T13: synthetic but CONTROLLED validation dataset.

We cannot rely on a blocked external tarball (CASF-2016 / PDBbind return
403 to scripted requests and require a browser). So the reproducible
verification uses a fixture with KNOWN RMSD + raw_score + controlled
noise (fixed seed). This satisfies SDD's "isolated, reproducible test"
rule: every AC is checked deterministically, not on a flaky download.

The REAL datasets (CASF-2016, PDBbind v2016 core, PoseBusters,
DockGen) are documented in RESEARCH.md with their official URLs and a
note that they need manual browser download. `validate --dataset casf2016`
is a documented future step, not assumed here.
"""
from __future__ import annotations

import numpy as np

from ..parse import NEAR_NATIVE_THRESHOLD, Pose


def make_fixture(seed: int = 42, n_systems: int = 20,
                poses_per_system: int = 10) -> tuple[list[Pose], list[Pose]]:
    """Return (all_poses, train_poses) with KNOWN ground truth.

    Construction (honest, controllable):
      - For each system, one pose is the "native-like" (RMSD ~ 1.0 A).
      - Others are decoys (RMSD 2.5 .. 6.0 A) drawn from a
        distribution.
      - raw_score (DiffDock-style confidence) is correlated with
        near-nativeness but NOISY: good poses get higher score, but the
        raw score is NOT calibrated (this is exactly the gap we fix).
      - train set = first half of systems (for platt calibration).
    """
    rng = np.random.default_rng(seed)
    all_poses: list[Pose] = []
    train_poses: list[Pose] = []

    for sys_i in range(n_systems):
        sid = f"sys_{sys_i:03d}"
        # native-like pose (near-native)
        native_rmsd = float(rng.uniform(0.6, 1.9))
        native_score = float(rng.normal(0.6, 0.25))  # confident but noisy
        all_poses.append(_mk(sid, 0, native_rmsd, native_score,
                               "diffdock_confidence"))
        # decoys
        for j in range(1, poses_per_system):
            r = float(rng.uniform(2.5, 6.0))
            # decoys get LOWER confidence on average, but with overlap
            s = float(rng.normal(-0.4, 0.6))
            all_poses.append(_mk(sid, j, r, s, "diffdock_confidence"))

        if sys_i < n_systems // 2:
            train_poses.extend(all_poses[-poses_per_system:])

    return all_poses, train_poses


def _mk(system_id, pose_id, rmsd, raw_score, source) -> Pose:
    nn = rmsd < NEAR_NATIVE_THRESHOLD
    return Pose(
        system_id=system_id,
        pose_id=pose_id,
        ligand_mol=None,  # fixture uses known RMSD directly
        raw_score=raw_score,
        score_source=source,
        rmsd=rmsd,
        near_native=nn,
    )
