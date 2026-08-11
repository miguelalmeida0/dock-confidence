# SPDX-FileCopyrightText: 2026 Pedro Sordo Martínez <amurlaniakea@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generate reproducible test fixtures (sample.sdf + sample_native.{sdf,pdb}).

Run from the repo root:  python3 tests/data/_make_samples.py
Writes into tests/data/ using paths relative to THIS file (no hardcoded
absolute paths — keeps the repo portable and CI-friendly).
"""
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

HERE = Path(__file__).resolve().parent
NATIVE_SDF = HERE / "sample_native.sdf"
NATIVE_PDB = HERE / "sample_native.pdb"
SAMPLE_SDF = HERE / "sample.sdf"

# Semilla fija -> reproducible. Fenol derivado.
smi = "Cc1ccccc1O"
base = Chem.MolFromSmiles(smi)
base = Chem.AddHs(base)
AllChem.EmbedMolecule(base, randomSeed=42)
AllChem.MMFFOptimizeMolecule(base)
base = Chem.RemoveHs(base)

# nativo: ligando cristalino optimizado (sin perturbacion)
w = Chem.SDWriter(str(NATIVE_SDF))
w.write(base)
w.close()
NATIVE_PDB.write_text(Chem.MolToPDBBlock(base))

# poses: copias con perturbaciones CRECIENTES y deterministas en atomos pesados
w = Chem.SDWriter(str(SAMPLE_SDF))
rng = np.random.default_rng(7)
for i in range(6):
    m = Chem.AddHs(base)
    conf = m.GetConformer()
    scale = 0.05 + 0.45 * i  # 0.05 .. 2.30 A de perturbacion
    for a in m.GetAtoms():
        if a.GetAtomicNum() > 1:
            pos = list(conf.GetAtomPosition(a.GetIdx()))
            pos[0] += float(rng.normal(0, scale))
            pos[1] += float(rng.normal(0, scale))
            pos[2] += float(rng.normal(0, scale))
            conf.SetAtomPosition(a.GetIdx(), pos)
    # NO MMFF relaxation: keep perturbed coords so RMSD grows with i
    # (poses 3-5 become true decoys > 2.0 A).
    m = Chem.RemoveHs(m)
    m.SetProp("minimizedAffinity", f"{-9.5 if i < 2 else -6.5 - i * 0.3:.3f}")
    m.SetProp("pose_id", str(i))
    w.write(m)
w.close()
print(f"OK fixtures written to {HERE}")
