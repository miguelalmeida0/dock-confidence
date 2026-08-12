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

ANTI-LEAKAGE GUARANTEE (AC7 fix):
  train_poses and test_poses are split BY SYSTEM_ID with ZERO overlap.
  Platt scaling is fit on train_poses only; every honest metric
  (ECE, top-1, ...) is evaluated on test_poses, never on the systems
  used to fit the model. An internal assert enforces this so the
  leakage regression can never silently return.
"""
from __future__ import annotations

from typing import List, NamedTuple

import numpy as np

from ..parse import NEAR_NATIVE_THRESHOLD, Pose


class FixtureData(NamedTuple):
    """Controlled synthetic split with an explicit, leak-free eval set.

    all_poses   : train_poses + test_poses (union; kept for inspection /
                  backward compatibility only — NEVER use it to evaluate a
                  model that was fit on train_poses).
    train_poses : systems used ONLY to fit Platt scaling.
    test_poses  : holdout; system_id set is DISJOINT from train_poses.
                  Use this for every honest evaluation metric.
    """
    all_poses: List[Pose]
    train_poses: List[Pose]
    test_poses: List[Pose]


def make_fixture(seed: int = 42, n_systems: int = 20,
                poses_per_system: int = 10) -> FixtureData:
    """Return a FixtureData with KNOWN ground truth and a leak-free split.

    Construction (honest, controllable):
      - For each system, one pose is the "native-like" (RMSD ~ 1.0 A).
      - Others are decoys (RMSD 2.5 .. 6.0 A) drawn from a
        distribution.
      - raw_score (DiffDock-style confidence) is correlated with
        near-nativeness but NOISY: good poses get higher score, but the
        raw score is NOT calibrated (this is exactly the gap we fix).
      - The split is BY SYSTEM: train = first half of systems,
        test = remaining half. This guarantees zero system_id overlap
        between the two sets (poses from a system used to fit Platt are
        never in the evaluation set).
    """
    rng = np.random.default_rng(seed)
    all_poses: List[Pose] = []
    train_poses: List[Pose] = []
    test_poses: List[Pose] = []

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

        # Split by SYSTEM, not by pose: first half of systems -> train,
        # second half -> test. Guarantees zero system_id overlap.
        block = all_poses[-poses_per_system:]
        if sys_i < n_systems // 2:
            train_poses.extend(block)
        else:
            test_poses.extend(block)

    # --- Regression guard: train and test must be disjoint by system_id ---
    train_sys = {p.system_id for p in train_poses}
    test_sys = {p.system_id for p in test_poses}
    overlap = train_sys & test_sys
    assert not overlap, (
        f"data leakage: train/test share system_id(s): {sorted(overlap)}"
    )
    # The two sets must partition all systems exactly once.
    assert train_sys | test_sys == {p.system_id for p in all_poses}, (
        "fixture split does not cover every system exactly once"
    )

    return FixtureData(all_poses=all_poses,
                       train_poses=train_poses,
                       test_poses=test_poses)


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
