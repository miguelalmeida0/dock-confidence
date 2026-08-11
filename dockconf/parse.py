# SPDX-FileCopyrightText: 2026 Pedro Sordo Martínez <amurlaniakea@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""FR1: Multi-dock pose parser -> list[Pose].

Reads DiffDock (.sdf multimodel), AutoDock-GPU/Smina (.sdf with
`minimizedAffinity` property), generic PDBQT (.pdbqt), and AutoDock
(.dlg). Extracts per pose: heavy-atom ligand, raw score + its source.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem

NEAR_NATIVE_THRESHOLD = 2.0  # Angstrom, standard in literature (AgenticPosesRanker SS3.6.3/3.7.1)


@dataclass
class Pose:
    """Single source of truth for one docking pose. Defined once, imported everywhere."""
    system_id: str
    pose_id: int
    ligand_mol: Optional[Chem.Mol] = None      # RDKit Mol (heavy atoms)
    raw_score: Optional[float] = None            # minimizedAffinity (Smina) or confidence (DiffDock)
    score_source: Optional[str] = None          # "smina_affinity" | "diffdock_confidence" | "pdbqt_score"
    rmsd: Optional[float] = None               # None if native not provided
    near_native: Optional[bool] = None          # rmsd < 2.0
    p_near_native: Optional[float] = None      # calibrated P(RMSD<2.0), set by calibrate.py
    confidence_band: Optional[str] = None       # "high"|"moderate"|"low"
    is_decoy: Optional[bool] = None            # set by report.filter_decoys


# ---------------------------------------------------------------------------
# DiffDock / Smina: multimodel SDF
# ---------------------------------------------------------------------------
def _sdf_props_to_score(mol: Chem.Mol) -> tuple[Optional[float], Optional[str]]:
    """Extract raw score + source from an SDF pose.

    DiffDock confidence lives in a property; Smina/AutoDock-GPU store
    `minimizedAffinity`. Returns (score, source).
    """
    if mol is None:
        return None, None
    # Smina / AutoDock-GPU empirical score
    for key in ("minimizedAffinity", "minimized_affinity", "affinity", "min_affinity"):
        if mol.HasProp(key):
            try:
                return float(mol.GetProp(key)), "smina_affinity"
            except ValueError:
                pass
    # DiffDock confidence (string like "0.42" or "-1.2")
    for key in ("confidence", "conf", "dock_confidence", "score"):
        if mol.HasProp(key):
            try:
                return float(mol.GetProp(key)), "diffdock_confidence"
            except ValueError:
                pass
    return None, None


def read_sdf(path: str, system_id: Optional[str] = None) -> list[Pose]:
    """Read a multimodel SDF (DiffDock or Smina output) into Pose list."""
    suppl = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
    poses: list[Pose] = []
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.GetSSSR(mol)
        except Exception:
            pass
        sid = system_id or (mol.GetProp("_Name") if mol.HasProp("_Name") else path)
        score, source = _sdf_props_to_score(mol)
        poses.append(Pose(system_id=str(sid), pose_id=i, ligand_mol=mol,
                         raw_score=score, score_source=source))
    return poses


# ---------------------------------------------------------------------------
# AutoDock PDBQT
# ---------------------------------------------------------------------------
def read_pdbqt(path: str, system_id: Optional[str] = None) -> list[Pose]:
    """Parse an AutoDock .pdbqt (one MODEL per pose)."""
    from rdkit.Chem import rdmolfiles

    block_lines: list[str] = []
    poses: list[Pose] = []
    pid = 0
    sid = system_id or path
    with open(path) as fh:
        for line in fh:
            if line.startswith("MODEL"):
                block_lines = []
            elif line.startswith("ENDMDL"):
                mol = rdmolfiles.MolFromPDBBlock("\n".join(block_lines),
                                                 removeHs=False, sanitize=False)
                score, source = _pdbqt_score(block_lines)
                poses.append(Pose(system_id=str(sid), pose_id=pid, ligand_mol=mol,
                                 raw_score=score, score_source=source))
                pid += 1
                block_lines = []
            else:
                block_lines.append(line.rstrip("\n"))
    return poses


def _pdbqt_score(block: list[str]) -> tuple[Optional[float], Optional[str]]:
    """AutoDock/Vina PDBQT stores 'REMARK VINA RESULT: affinity ...'."""
    for line in block:
        if "VINA RESULT" in line or "REMARK VINA" in line:
            # grab the first float token (the affinity, kcal/mol)
            for tok in line.replace(":", " ").split():
                try:
                    return float(tok), "pdbqt_score"
                except ValueError:
                    continue
    return None, None


# ---------------------------------------------------------------------------
# AutoDock .dlg (log file)
# ---------------------------------------------------------------------------
def read_dlg(path: str, system_id: Optional[str] = None) -> list[Pose]:
    """Parse AutoDock .dlg: clusters with 'RANKING' affinity lines."""
    poses: list[Pose] = []
    sid = system_id or path
    pid = 0
    with open(path) as fh:
        for line in fh:
            if "RANKING" in line and "ESTIMATED" in line:
                # RANKING 1  CLUSTER 1    -9.50       ESTIMATED AFFINITY
                parts = line.split()
                # affinity is the only float token before ESTIMATED
                try:
                    aff = next(float(tok) for tok in reversed(parts)
                               if tok.replace(".", "", 1).replace("-", "", 1).isdigit())
                    poses.append(Pose(system_id=str(sid), pose_id=pid,
                                     raw_score=aff, score_source="pdbqt_score"))
                    pid += 1
                except (ValueError, StopIteration):
                    pass
    return poses


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def read_poses(path: str, fmt: Optional[str] = None,
              system_id: Optional[str] = None) -> list[Pose]:
    """Auto-detect format or use explicit fmt in {sdf, pdbqt, dlg}."""
    if fmt is None:
        if path.endswith(".sdf") or path.endswith(".mol2"):
            fmt = "sdf"
        elif path.endswith(".pdbqt"):
            fmt = "pdbqt"
        elif path.endswith(".dlg"):
            fmt = "dlg"
        else:
            fmt = "sdf"
    if fmt == "sdf":
        return read_sdf(path, system_id)
    if fmt == "pdbqt":
        return read_pdbqt(path, system_id)
    if fmt == "dlg":
        return read_dlg(path, system_id)
    raise ValueError(f"Unknown format: {fmt}")
