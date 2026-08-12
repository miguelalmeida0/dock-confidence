"""FR3: CLI entrypoint.

Subcommands:
  parse     read a dock output into poses (JSON summary)
  calibrate apply calibration (heuristic or platt) -> JSON+PNG
  validate  run on a dataset (fixture or --dataset) -> ECE report
  report    recompute decoy filter + reliability diagram from a pose JSON
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .parse import read_poses
from .rmsd import annotate_rmsd
from .calibrate import calibrate
from .metrics import (
    expected_calibration_error,
    raw_score_ece,
    top1_accuracy,
    random_baseline_accuracy,
)
from .report import write_json, reliability_diagram, filter_decoys


def _load_poses(json_path: str):
    with open(json_path) as fh:
        data = json.load(fh)
    # re-hydrate minimal Pose objects for reporting
    from .parse import Pose
    return [Pose(**{k: v for k, v in d.items()}) for d in data]


def cmd_parse(args):
    poses = read_poses(args.input, fmt=args.format, system_id=args.system)
    if args.native:
        poses = annotate_rmsd(poses, args.native)
    if args.calibrate_mode:
        poses = calibrate(poses, mode=args.calibrate_mode,
                        train=_load_poses(args.train) if args.train else None)
    out = args.out or (str(Path(args.input).with_suffix("")) + ".poses.json")
    write_json(poses, out)
    n_native = sum(1 for p in poses if p.near_native is True)
    print(f"[parse] {len(poses)} poses from {args.input}")
    print(f"[parse] near-native (RMSD<2.0A): {n_native}")
    print(f"[parse] wrote {out}")
    return 0


def cmd_calibrate(args):
    poses = read_poses(args.input, fmt=args.format, system_id=args.system)
    if args.native:
        poses = annotate_rmsd(poses, args.native)
    poses = calibrate(poses, mode=args.mode,
                      train=_load_poses(args.train) if args.train else None)
    out = args.out or (str(Path(args.input).with_suffix("")) + ".calibrated.json")
    write_json(poses, out)
    png = args.diagram or (str(Path(out).with_suffix("")) + ".reliability.png")
    reliability_diagram(poses, png)
    ece = expected_calibration_error(poses)
    print(f"[calibrate] mode={args.mode} poses={len(poses)}")
    print(f"[calibrate] ECE(calibrated) = {ece:.4f}")
    print(f"[calibrate] wrote {out}")
    print(f"[calibrate] diagram {png}")
    return 0


def cmd_validate(args):
    if args.fixture:
        from .data.fixture import make_fixture
        poses, train = make_fixture(seed=args.seed, n_systems=args.n_systems)
    else:
        poses = read_poses(args.input, fmt=args.format)
        if args.native:
            poses = annotate_rmsd(poses, args.native)
        train = poses if args.train is None else _load_poses(args.train)
    if args.mode == "platt" and train is not None:
        poses = calibrate(poses, mode="platt", train=train)
    else:
        poses = calibrate(poses, mode=args.mode)
    ece_cal = expected_calibration_error(poses)
    ece_raw = raw_score_ece(poses)
    acc = top1_accuracy(poses)
    rnd = random_baseline_accuracy(poses)
    print("=" * 52)
    print(" dock-confidence :: validation report")
    print("=" * 52)
    print(f" poses evaluated      : {len(poses)}")
    print(f" ECE (calibrated)   : {ece_cal:.4f}")
    print(f" ECE (raw score)     : {ece_raw:.4f}")
    print(f" top-1 accuracy      : {acc:.3f}")
    print(f" random baseline     : {rnd:.3f}")
    if ece_cal is not None and ece_raw is not None:
        verdict = "PASS (calibrated < raw)" if ece_cal < ece_raw else "CHECK (no improvement)"
        print(f" AC7 verdict         : {verdict}")
    print("=" * 52)
    return 0


def cmd_report(args):
    poses = _load_poses(args.input)
    poses = filter_decoys(poses, threshold=args.threshold)
    out = args.out or args.input
    write_json(poses, out)
    png = args.diagram or (str(Path(out).with_suffix("")) + ".reliability.png")
    reliability_diagram(poses, png)
    n_decoy = sum(1 for p in poses if p.is_decoy is True)
    print(f"[report] decoys flagged (p<{args.threshold}): {n_decoy}")
    print(f"[report] wrote {out}")
    print(f"[report] diagram {png}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dock-confidence",
                                description="Calibrated confidence for docking poses.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("parse", help="read dock output -> poses JSON")
    sp.add_argument("--input", required=True)
    sp.add_argument("--format", choices=["sdf", "pdbqt", "dlg"], default=None)
    sp.add_argument("--system", default=None)
    sp.add_argument("--native", default=None, help="native PDB/SDF for RMSD")
    sp.add_argument("--calibrate-mode", choices=["heuristic", "platt"], default=None)
    sp.add_argument("--train", default=None)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_parse)

    sc = sub.add_parser("calibrate", help="calibrate + report")
    sc.add_argument("--input", required=True)
    sc.add_argument("--format", choices=["sdf", "pdbqt", "dlg"], default=None)
    sc.add_argument("--native", default=None)
    sc.add_argument("--mode", choices=["heuristic", "platt"], default="heuristic")
    sc.add_argument("--train", default=None)
    sc.add_argument("--out", default=None)
    sc.add_argument("--diagram", default=None)
    sc.set_defaults(func=cmd_calibrate)

    sv = sub.add_parser("validate", help="validation report on a dataset")
    sv.add_argument("--fixture", action="store_true", help="use built-in synthetic dataset")
    sv.add_argument("--seed", type=int, default=42)
    sv.add_argument("--n-systems", type=int, default=20)
    sv.add_argument("--input", default=None)
    sv.add_argument("--format", choices=["sdf", "pdbqt", "dlg"], default=None)
    sv.add_argument("--native", default=None)
    sv.add_argument("--train", default=None)
    sv.add_argument("--mode", choices=["heuristic", "platt"], default="heuristic")
    sv.set_defaults(func=cmd_validate)

    sr = sub.add_parser("report", help="decoy filter + reliability diagram")
    sr.add_argument("--input", required=True)
    sr.add_argument("--threshold", type=float, default=0.5)
    sr.add_argument("--out", default=None)
    sr.add_argument("--diagram", default=None)
    sr.set_defaults(func=cmd_report)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
