#!/usr/bin/env python3
"""Fail closed when calibration and reported ECE share fixture systems."""

from __future__ import annotations

import json
from pathlib import Path


CLI = Path("dockconf/cli.py")
FIXTURE = Path("dockconf/data/fixture.py")
TEST = Path("tests/test_dockconf.py")
cli = CLI.read_text(encoding="utf-8")
fixture = FIXTURE.read_text(encoding="utf-8")
test = TEST.read_text(encoding="utf-8")

requirements = [
    ("fx.test_poses", cli, CLI, "validation must evaluate the disjoint fixture holdout"),
    ("train=fx.train_poses", test, TEST, "calibration must fit on fixture training systems"),
    ("test_poses", fixture, FIXTURE, "the fixture must expose a distinct test partition"),
    ("assert train_systems.isdisjoint(test_systems)", test, TEST, "the suite must assert system disjointness"),
]
findings = [
    {
        "rule_id": "CHRONOS-DISJOINT-EVALUATION-001",
        "path": str(path),
        "line": 1,
        "reason": reason,
    }
    for needle, source, path, reason in requirements
    if needle not in source
]

payload = {
    "schema": "chronos.ci-evaluation.v1",
    "policy_id": "chronos-ext013-disjoint-calibration-evaluation",
    "policy_state": "installed",
    "status": "FAIL" if findings else "PASS",
    "blocking_findings": findings,
    "coverage": {
        "mode": "COMPLETE",
        "eligible_supported_files": 3,
        "analyzed_supported_files": 3,
        "missing_supported_files": [],
    },
}
print(json.dumps(payload, indent=2, sort_keys=True))
for finding in findings:
    print(
        f"::error file={finding['path']},line={finding['line']}::"
        f"{finding['rule_id']}: {finding['reason']}"
    )
raise SystemExit(2 if findings else 0)
