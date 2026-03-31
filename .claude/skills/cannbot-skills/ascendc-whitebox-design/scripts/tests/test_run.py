import json
import os
import subprocess
import sys
import tempfile
import pytest

CWD = os.path.join(os.path.dirname(__file__), "..")

SAMPLE_FLAT = {
    "params": {"a": [1, 2], "b": ["x", "y"]},
    "coverage": "medium",
}

SAMPLE_GROUPED = {
    "groups": [
        {"id": "g1", "params": {"a": [1, 2], "b": ["x", "y"]}},
        {"id": "g2", "params": {"c": [True, False]}},
    ],
    "coverage": "high",
}


def _run(dims_dict, extra_args=None):
    tmp = tempfile.mkdtemp()
    dims_path = os.path.join(tmp, "param_def.json")
    with open(dims_path, "w") as f:
        json.dump(dims_dict, f)
    cmd = [sys.executable, "run.py",
           "--param-def", dims_path,
           "--output_dir", tmp,
           "--seed", "42"]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=CWD)
    params_path = os.path.join(tmp, "cases.json")
    params = None
    if os.path.exists(params_path):
        with open(params_path) as f:
            params = json.load(f)
    return result.returncode, params, result.stdout


def test_run_flat():
    rc, params, _ = _run(SAMPLE_FLAT)
    assert rc == 0
    assert params is not None
    assert len(params) > 0


def test_run_grouped():
    rc, params, _ = _run(SAMPLE_GROUPED)
    assert rc == 0
    assert len(params) > 0
    groups = {p["_group"] for p in params}
    assert "g1" in groups
    assert "g2" in groups


def test_run_coverage_override():
    rc, params, _ = _run(SAMPLE_FLAT, ["--coverage", "high"])
    assert rc == 0
    assert len(params) == 4  # 2x2 cartesian


def test_run_prints_summary():
    rc, _, stdout = _run(SAMPLE_FLAT)
    assert rc == 0
    assert "combinations" in stdout.lower() or "parameter" in stdout.lower()


def test_run_invalid_json():
    tmp = tempfile.mkdtemp()
    bad_path = os.path.join(tmp, "param_def.json")
    with open(bad_path, "w") as f:
        f.write("not json")
    cmd = [sys.executable, "run.py", "--param-def", bad_path, "--output_dir", tmp]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=CWD)
    assert result.returncode != 0


def test_run_with_constraints():
    """E2E: constraints should filter invalid combos through CLI."""
    spec = {
        "groups": [
            {
                "id": "constrained",
                "params": {
                    "dst_type": [2, 34, 40],
                    "round_mode": ["rint", "round", "floor"],
                },
                "constraints": [
                    {"if": {"dst_type": [2]}, "then": {"round_mode": ["rint"]}},
                    {"if": {"dst_type": [34]}, "then": {"round_mode": ["round"]}},
                ],
            }
        ],
        "coverage": "high",
    }
    rc, params, _ = _run(spec)
    assert rc == 0
    assert params is not None
    for p in params:
        if p["dst_type"] == 2:
            assert p["round_mode"] == "rint", f"constraint violated: {p}"
        if p["dst_type"] == 34:
            assert p["round_mode"] == "round", f"constraint violated: {p}"
    # dst_type=40 has no constraint, all modes valid
    modes_40 = {p["round_mode"] for p in params if p["dst_type"] == 40}
    assert modes_40 == {"rint", "round", "floor"}


def test_run_validation_missing_id():
    spec = {"groups": [{"params": {"a": [1]}}]}
    rc, _, _ = _run(spec)
    assert rc != 0

def test_run_validation_empty_dim():
    spec = {"groups": [{"id": "g1", "params": {"a": []}}]}
    rc, _, _ = _run(spec)
    assert rc != 0

def test_run_validation_bad_constraint():
    spec = {"groups": [{"id": "g1", "params": {"a": [1]}, "constraints": [123]}]}
    rc, _, _ = _run(spec)
    assert rc != 0

def test_run_validation_bad_desc_rule():
    spec = {"groups": [{"id": "g1", "params": {"a": [1]}, "desc_rules": [{"formula": "a > 0"}]}]}
    rc, _, _ = _run(spec)
    assert rc != 0  # missing 'desc' field
