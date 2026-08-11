# SPDX-FileCopyrightText: 2026 Pedro Sordo Martínez <amurlaniakea@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Edge-case fixtures: real PDBQT + AutoDock .dlg sample text.

These are synthetic but structurally valid, written once so the parser
tests are reproducible without external downloads.
"""
PDBQT_TEXT = """MODEL
REMARK  VINA RESULT: -9.50 kcal/mol, dist=0.0, rmsd=0.0
REMARK  Pose 0
ATOM      1  C   UNL     1       0.000   0.000   0.000  0.00  0.00    +0    -0.10
ATOM      2  C   UNL     1       1.390   0.000   0.000  0.00  0.00    +0    -0.10
ATOM      3  C   UNL     1       2.085   1.205   0.000  0.00  0.00    +0    -0.10
ATOM      4  C   UNL     1       1.390   2.410   0.000  0.00  0.00    +0    -0.10
ATOM      5  C   UNL     1       0.000   2.410   0.000  0.00  0.00    +0    -0.10
ATOM      6  C   UNL     1      -0.695   1.205   0.000  0.00  0.00    +0    -0.10
ATOM      7  O   UNL     1       3.400   1.205   0.000  0.00  0.00    +0    -0.30
TER
ENDMDL
"""

DLG_TEXT = """________________________________________________________________________________
AutoDock 4.2.6 job 1
________________________________________________________________________________
INPUT PDBQT FILE: ligand.pdbqt
________________________________________________________________________________

________________________________________________________________________________
RANKING 1  CLUSTER 1    -9.50       ESTIMATED AFFINITY
RANKING 2  CLUSTER 2    -8.10       ESTIMATED AFFINITY
________________________________________________________________________________
"""


def write_pdbqt(path: str) -> None:
    with open(path, "w") as f:
        f.write(PDBQT_TEXT)


def write_dlg(path: str) -> None:
    with open(path, "w") as f:
        f.write(DLG_TEXT)
