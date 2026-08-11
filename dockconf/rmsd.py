# SPDX-FileCopyrightText: 2026 Pedro Sordo Martínez <amurlaniakea@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""FR2: heavy-atom symmetric RMSD to native (RDKit AlignMol/GetBestRMS).

Ground-truth definition follows AgenticPosesRanker Eq.10 / SS3.7.1:
the best pose is the one with lowest heavy-atom symmetry-corrected
RMSD to the crystallographic binding mode. Threshold RMSD < 2.0 A
marks a near-native pose (SS3.6.3, SS3.7.1).
"""
from __future__ import annotations

from typing import Optional

from rdkit import Chem
from rdkit.Chem import rdMolAlign

from .parse import NEAR_NATIVE_THRESHOLD, Pose


def _heavy_atom_map(mol: Chem.Mol) -> list[int]:
    """Indices of Heavy (non-H) atoms, in order, for both mols."""
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]


def rmsd_to_native(pose_mol: Chem.Mol, native_mol: Chem.Mol) -> Optional[float]:
    """Heavy-atom RMSD (symmetric) to the native crystal ligand.

    Returns None if alignment is invalid (does NOT crash). Parity with
    'heavy-atom symmetry-corrected RMSD' from the literature.

    Both molecules are sanitized so AlignMol's sub-structure match works
    (required by RDKit regardless of atom order when a map is given).
    """
    if pose_mol is None or native_mol is None:
        return None
    try:
        pm = Chem.Mol(pose_mol)
        nm = Chem.Mol(native_mol)
        Chem.SanitizeMol(pm)
        Chem.SanitizeMol(nm)
    except Exception:
        return None

    # Strategy 1: heavy-atom map via MCS (handles symmetry +
    # different atom ordering between predicted ligand and crystal).
    try:
        atom_map = _canonical_map(pm, nm)
        if atom_map:
            rmsd = rdMolAlign.AlignMol(pm, nm, atomMap=atom_map)
            return float(rmsd)
        # No full heavy-atom MCS match -> DIFFERENT molecule.
        # Return None instead of a bogus 0.0 from a blind AlignMol.
        return None
    except Exception:
        pass

    # Strategy 2: raw AlignMol (same ordering, fallback).
    try:
        rmsd = rdMolAlign.AlignMol(pm, nm)
        return float(rmsd)
    except Exception:
        return None


def _canonical_map(prb: Chem.Mol, ref: Chem.Mol) -> list[tuple[int, int]]:
    """Heavy-atom match via Maximum Common Substructure (topology-aware).

    Handles symmetry + different atom ordering robustly: the predicted
    ligand and the crystal native are the SAME molecule, so an MCS
    match yields a valid 1:1 heavy-atom map. Returns [] if no
    full heavy-atom MCS match (different molecule -> bail).
    """
    from rdkit.Chem import rdFMCS

    res = rdFMCS.FindMCS(
        [prb, ref],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        completeRingsOnly=True,
        ringMatchesRingOnly=True,
    )
    if res.canceled or res.numAtoms == 0:
        return []
    # Require the MCS to cover ALL heavy atoms of BOTH molecules.
    # If not, the two ligands are different molecules -> no valid map.
    n_prb = sum(1 for a in prb.GetAtoms() if a.GetAtomicNum() > 1)
    n_ref = sum(1 for a in ref.GetAtoms() if a.GetAtomicNum() > 1)
    if res.numAtoms < min(n_prb, n_ref):
        return []
    pattern = Chem.MolFromSmarts(res.smartsString)
    if pattern is None:
        return []
    pm = prb.GetSubstructMatch(pattern)
    rm = ref.GetSubstructMatch(pattern)
    if len(pm) != len(rm) or len(pm) == 0:
        return []
    return [(int(i), int(j)) for i, j in zip(pm, rm)]


def annotate_rmsd(poses: list[Pose], native_path: str,
                   system_match: bool = True) -> list[Pose]:
    """Given a native PDB/SDF, compute rmsd + near_native for each Pose.

    The MCS alignment between a predicted ligand and its crystal native
    is computed ONCE per `system_id` (native is shared within a
    system) and cached, so N poses cost 1 MCS + N cheap AlignMol
    calls instead of N expensive MCS computations.
    """
    native = _load_native(native_path)
    if native is None:
        return poses
    # Cache: system_id -> (sanitized native Mol, precomputed MCS atom_map)
    # The MCS map depends only on topology (identical across a system's
    # poses), so we compute it ONCE per system, not per pose.
    _cache: dict[str, tuple[Chem.Mol, Optional[list[tuple[int, int]]]]] = {}

    for p in poses:
        if p.ligand_mol is None:
            p.rmsd = None
            p.near_native = None
            continue
        sid = p.system_id
        if sid not in _cache:
            nm = Chem.Mol(native)
            try:
                Chem.SanitizeMol(nm)
            except Exception:
                pass
            # compute MCS map once, using this first pose as reference.
            # Both reference pose and native must be sanitized so the MCS
            # matches topology consistently (aromaticity, valence).
            ref = Chem.Mol(p.ligand_mol)
            try:
                Chem.SanitizeMol(ref)
            except Exception:
                pass
            amap = _canonical_map(ref, nm)
            _cache[sid] = (nm, amap)
        nm, atom_map = _cache[sid]
        if atom_map is None:
            p.rmsd = None
            p.near_native = None
            continue
        r = _rmsd_with_map(p.ligand_mol, nm, atom_map)
        p.rmsd = r
        p.near_native = (r is not None and r < NEAR_NATIVE_THRESHOLD)
    return poses


def _rmsd_with_map(pose_mol: Chem.Mol, native_mol: Chem.Mol,
                   atom_map: list[tuple[int, int]]) -> Optional[float]:
    """Align a pose to a native using a precomputed heavy-atom map."""
    if pose_mol is None or native_mol is None or not atom_map:
        return None
    try:
        pm = Chem.Mol(pose_mol)
        Chem.SanitizeMol(pm)
        return float(rdMolAlign.AlignMol(pm, native_mol, atomMap=atom_map))
    except Exception:
        return None


def _load_native(path: str) -> Optional[Chem.Mol]:
    try:
        if path.endswith((".pdb", ".pdbqt")):
            mol = Chem.MolFromPDBFile(path, removeHs=False, sanitize=False)
        elif path.endswith((".sdf", ".mol2", ".mol")):
            mol = Chem.MolFromMolFile(path, removeHs=False, sanitize=False)
        else:
            mol = Chem.MolFromPDBFile(path, removeHs=False, sanitize=False)
    except Exception:
        return None
    if mol is None:
        return None
    return mol
