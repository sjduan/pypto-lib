"""Tests for engine.enumerator module."""

import itertools
import json
import os
import tempfile

import pytest

from engine.enumerator import expand_dimension, enumerate_params, enumerate_params_from_file


# ---------------------------------------------------------------------------
# expand_dimension tests
# ---------------------------------------------------------------------------

class TestExpandDimension:
    def test_branch_split(self):
        dim_def = {
            "thresholds": [{"type": "branch_split", "value": 10}],
            "min": 1,
            "max": 100,
        }
        result = expand_dimension(dim_def)
        assert 9 in result   # v-1
        assert 10 in result  # v
        assert 11 in result  # v+1

    def test_alignment(self):
        dim_def = {
            "thresholds": [{"type": "alignment", "value": 32}],
            "min": 1,
            "max": 200,
        }
        result = expand_dimension(dim_def)
        # k*v-1, k*v for k=1..3 → 31,32, 63,64, 95,96
        assert 31 in result
        assert 32 in result
        assert 63 in result
        assert 64 in result
        assert 95 in result
        assert 96 in result

    def test_divisor(self):
        dim_def = {
            "thresholds": [{"type": "divisor", "value": 16}],
            "min": 1,
            "max": 100,
        }
        result = expand_dimension(dim_def)
        # k*v, k*v+1 for k=1..3 → 16,17, 32,33, 48,49
        assert 16 in result
        assert 17 in result
        assert 32 in result
        assert 33 in result
        assert 48 in result
        assert 49 in result

    def test_min_max_filtering(self):
        dim_def = {
            "thresholds": [{"type": "branch_split", "value": 5}],
            "min": 5,
            "max": 10,
        }
        result = expand_dimension(dim_def)
        # v-1=4 should be filtered out (below min)
        assert 4 not in result
        assert 5 in result
        assert 6 in result

    def test_alignment_filtering(self):
        """Values filtered by alignment constraint."""
        dim_def = {
            "thresholds": [{"type": "branch_split", "value": 16}],
            "min": 1,
            "max": 100,
            "alignment": 8,
        }
        result = expand_dimension(dim_def)
        for v in result:
            assert v % 8 == 0, f"{v} is not aligned to 8"

    def test_random_count(self):
        dim_def = {
            "thresholds": [{"type": "branch_split", "value": 50}],
            "min": 1,
            "max": 100,
            "random_count": 5,
        }
        result = expand_dimension(dim_def, seed=42)
        # Should have boundary values + up to 5 random interior values
        assert len(result) >= 3  # at least the boundary values
        # All values within range
        for v in result:
            assert 1 <= v <= 100

    def test_plain_list_passthrough(self):
        values = ["float16", "bfloat16", "float32"]
        result = expand_dimension(values)
        assert result == values


# ---------------------------------------------------------------------------
# enumerate_params tests
# ---------------------------------------------------------------------------

class TestEnumerateParams:
    def test_low_one_at_a_time(self):
        spec = {
            "params": {
                "dtype": ["float16", "bfloat16"],
                "size": [10, 20, 30],
            },
            "coverage": "low",
        }
        cases = enumerate_params(spec)
        # Every value of every dimension must appear at least once
        dtypes_seen = {c["dtype"] for c in cases}
        sizes_seen = {c["size"] for c in cases}
        assert dtypes_seen == {"float16", "bfloat16"}
        assert sizes_seen == {10, 20, 30}

    def test_medium_pairwise(self):
        spec = {
            "params": {
                "a": [1, 2],
                "b": [3, 4],
                "c": [5, 6],
            },
            "coverage": "medium",
        }
        cases = enumerate_params(spec)
        # Every pair of values from any two params must appear
        dim_names = list(spec["params"].keys())
        for i, j in itertools.combinations(range(len(dim_names)), 2):
            di, dj = dim_names[i], dim_names[j]
            required_pairs = set(
                itertools.product(spec["params"][di], spec["params"][dj])
            )
            covered_pairs = {(c[di], c[dj]) for c in cases}
            assert required_pairs <= covered_pairs, (
                f"Missing pairs for ({di},{dj}): {required_pairs - covered_pairs}"
            )

    def test_high_cartesian(self):
        spec = {
            "params": {
                "a": [1, 2],
                "b": [3, 4],
                "c": [5, 6],
            },
            "coverage": "high",
        }
        cases = enumerate_params(spec)
        assert len(cases) == 2 * 2 * 2

    def test_with_threshold_dimension(self):
        spec = {
            "params": {
                "dtype": ["float16", "float32"],
                "size": {
                    "thresholds": [{"type": "branch_split", "value": 10}],
                    "min": 1,
                    "max": 100,
                },
            },
            "coverage": "low",
        }
        cases = enumerate_params(spec)
        sizes_seen = {c["size"] for c in cases}
        # Should include boundary values from threshold expansion
        assert 9 in sizes_seen
        assert 10 in sizes_seen
        assert 11 in sizes_seen

    def test_empty_params(self):
        spec = {"params": {}, "coverage": "low"}
        cases = enumerate_params(spec)
        # Empty params → one empty case or no cases
        assert cases == [{}] or cases == []

    def test_single_dimension(self):
        spec = {
            "params": {"x": [1, 2, 3]},
            "coverage": "low",
        }
        cases = enumerate_params(spec)
        values_seen = {c["x"] for c in cases}
        assert values_seen == {1, 2, 3}

    def test_coverage_override(self):
        """coverage kwarg overrides spec's coverage."""
        spec = {
            "params": {
                "a": [1, 2],
                "b": [3, 4],
            },
            "coverage": "low",
        }
        cases = enumerate_params(spec, coverage="high")
        assert len(cases) == 4  # full cartesian


class TestEnumerateParamsFromFile:
    def test_loads_json(self, tmp_path):
        spec = {
            "params": {"x": [1, 2]},
            "coverage": "low",
        }
        path = tmp_path / "spec.json"
        path.write_text(json.dumps(spec))
        cases = enumerate_params_from_file(str(path))
        assert {c["x"] for c in cases} == {1, 2}


class TestEnumerateParamsGrouped:
    """Tests for grouped params."""

    def test_grouped_basic(self):
        spec = {
            "groups": [
                {"id": "path_a", "params": {"x": [1, 2], "y": ["a", "b"]}},
                {"id": "path_b", "params": {"x": [3, 4], "z": [True, False]}},
            ],
            "coverage": "high",
        }
        params = enumerate_params(spec)
        path_a = [p for p in params if p.get("_group") == "path_a"]
        path_b = [p for p in params if p.get("_group") == "path_b"]
        assert len(path_a) == 4  # 2x2 cartesian
        assert len(path_b) == 4  # 2x2 cartesian
        assert len(params) == 8

    def test_grouped_has_group_field(self):
        spec = {
            "groups": [
                {"id": "grp1", "params": {"a": [1]}},
            ],
            "coverage": "low",
        }
        params = enumerate_params(spec)
        assert all(p["_group"] == "grp1" for p in params)

    def test_grouped_independent_params(self):
        """Each group can have completely different dimension names."""
        spec = {
            "groups": [
                {"id": "static", "params": {"x_dtype": ["int32"], "bias": [True, False]}},
                {"id": "dynamic", "params": {"x_dtype": ["fp16", "bf16"], "qscale": [True, False]}},
            ],
            "coverage": "high",
        }
        params = enumerate_params(spec)
        static = [p for p in params if p["_group"] == "static"]
        dynamic = [p for p in params if p["_group"] == "dynamic"]
        assert len(static) == 2  # 1 x 2
        assert len(dynamic) == 4  # 2 x 2
        # static cases don't have qscale, dynamic don't have bias
        assert "bias" in static[0]
        assert "qscale" not in static[0]
        assert "qscale" in dynamic[0]
        assert "bias" not in dynamic[0]

    def test_grouped_pairwise(self):
        spec = {
            "groups": [
                {"id": "g1", "params": {"a": [1, 2], "b": [3, 4], "c": [5, 6]}},
            ],
            "coverage": "medium",
        }
        params = enumerate_params(spec)
        # Pairwise should cover all pairs
        pairs_ab = {(p["a"], p["b"]) for p in params}
        assert (1, 3) in pairs_ab
        assert (2, 4) in pairs_ab


class TestConstraintFiltering:
    """Tests for constraint-based filtering."""

    def test_if_then_constraint(self):
        spec = {
            "params": {"dst_type": [2, 34, 40], "round_mode": ["rint", "round", "floor"]},
            "constraints": [
                {"if": {"dst_type": [2]}, "then": {"round_mode": ["rint"]}},
                {"if": {"dst_type": [34]}, "then": {"round_mode": ["round"]}},
            ],
            "coverage": "high",
        }
        params = enumerate_params(spec)
        for p in params:
            if p["dst_type"] == 2:
                assert p["round_mode"] == "rint"
            if p["dst_type"] == 34:
                assert p["round_mode"] == "round"
        # dst_type=40 has no constraint, all round_modes valid
        modes_40 = {p["round_mode"] for p in params if p["dst_type"] == 40}
        assert modes_40 == {"rint", "round", "floor"}

    def test_requires_constraint(self):
        spec = {
            "params": {"x_dtype": ["int32", "fp16"], "quant_mode": ["static", "dynamic"]},
            "constraints": [
                {"requires": {"x_dtype": "int32", "quant_mode": "static"}},
                {"requires": {"x_dtype": "fp16", "quant_mode": "dynamic"}},
            ],
            "coverage": "high",
        }
        params = enumerate_params(spec)
        for p in params:
            if p["x_dtype"] == "int32":
                assert p["quant_mode"] == "static"
            if p["x_dtype"] == "fp16":
                assert p["quant_mode"] == "dynamic"

    def test_text_constraints_ignored(self):
        """Text-only constraints should be skipped, not cause errors."""
        spec = {
            "params": {"a": [1, 2]},
            "constraints": [
                "outDimy = x_last / 2",
                "this is just documentation",
            ],
            "coverage": "high",
        }
        params = enumerate_params(spec)
        assert len(params) == 2

    def test_grouped_constraints(self):
        spec = {
            "groups": [
                {
                    "id": "g1",
                    "params": {"dst_type": [2, 40], "round_mode": ["rint", "floor"]},
                    "constraints": [{"if": {"dst_type": [2]}, "then": {"round_mode": ["rint"]}}],
                }
            ],
            "coverage": "high",
        }
        params = enumerate_params(spec)
        for p in params:
            if p["dst_type"] == 2:
                assert p["round_mode"] == "rint"

    def test_no_constraints_passes_all(self):
        spec = {"params": {"a": [1, 2], "b": [3, 4]}, "coverage": "high"}
        params = enumerate_params(spec)
        assert len(params) == 4


class TestLowConfigs:
    """Tests for low coverage with low_configs (network common shapes)."""

    def test_low_uses_low_configs(self):
        spec = {
            "groups": [
                {
                    "id": "g1",
                    "low_configs": [
                        {"x_dtype": "bf16", "x_last": 8192, "note": "LLaMA-7B"},
                        {"x_dtype": "bf16", "x_last": 10240, "note": "LLaMA-13B"},
                    ],
                    "params": {"x_dtype": ["bf16", "fp16"], "x_last": [100, 200]},
                }
            ],
            "coverage": "low",
        }
        params = enumerate_params(spec)
        assert len(params) == 2
        assert params[0]["x_last"] == 8192
        assert params[1]["x_last"] == 10240
        assert all(p["_group"] == "g1" for p in params)

    def test_medium_includes_low_configs_too(self):
        spec = {
            "groups": [
                {
                    "id": "g1",
                    "low_configs": [{"x_last": 8192}],
                    "params": {"a": [1, 2], "b": [3, 4]},
                }
            ],
            "coverage": "medium",
        }
        params = enumerate_params(spec)
        # medium uses params AND low_configs
        assert any("a" in p for p in params)
        assert any(p.get("x_last") == 8192 for p in params)

    def test_low_fallback_without_low_configs(self):
        spec = {
            "groups": [
                {"id": "g1", "params": {"a": [1, 2, 3]}},
            ],
            "coverage": "low",
        }
        params = enumerate_params(spec)
        # Falls back to one-at-a-time
        assert len(params) == 3

    def test_medium_includes_low_configs(self):
        spec = {
            "groups": [
                {
                    "id": "g1",
                    "low_configs": [{"a": 99, "b": 99, "note": "LLaMA"}],
                    "params": {"a": [1, 2], "b": [3, 4]},
                }
            ],
            "coverage": "medium",
        }
        params = enumerate_params(spec)
        # Should include both pairwise combos AND the low_config
        assert any(p.get("a") == 99 and p.get("b") == 99 for p in params)
        # Also has regular combos
        assert any(p.get("a") == 1 for p in params)

    def test_high_includes_low_configs(self):
        spec = {
            "groups": [
                {
                    "id": "g1",
                    "low_configs": [{"a": 99, "b": 99}],
                    "params": {"a": [1, 2], "b": [3, 4]},
                }
            ],
            "coverage": "high",
        }
        params = enumerate_params(spec)
        assert len(params) == 5  # 4 cartesian + 1 low_config
        assert any(p.get("a") == 99 for p in params)

    def test_low_configs_dedup_with_medium(self):
        """If a low_config matches a pairwise combo, should not duplicate."""
        spec = {
            "groups": [
                {
                    "id": "g1",
                    "low_configs": [{"a": 1, "b": 3}],
                    "params": {"a": [1, 2], "b": [3, 4]},
                }
            ],
            "coverage": "medium",
        }
        params = enumerate_params(spec)
        # {a:1, b:3} exists in pairwise already, should not appear twice
        matches = [p for p in params if p.get("a") == 1 and p.get("b") == 3]
        assert len(matches) == 1

    def test_low_configs_note_field_preserved(self):
        spec = {
            "groups": [
                {
                    "id": "g1",
                    "low_configs": [{"x_last": 8192, "note": "LLaMA-7B"}],
                    "params": {"x_last": [100]},
                }
            ],
            "coverage": "low",
        }
        params = enumerate_params(spec)
        assert params[0].get("note") == "LLaMA-7B"


class TestFormulaConstraint:
    """Tests for formula-based constraint filtering."""

    def test_simple_inequality(self):
        spec = {
            "params": {"k": [1, 2, 4, 8, 16], "expertCount": [8, 16]},
            "constraints": [{"formula": "k <= expertCount"}],
            "coverage": "high",
        }
        params = enumerate_params(spec)
        for p in params:
            assert p["k"] <= p["expertCount"], f"violated: {p}"

    def test_multi_variable_formula(self):
        spec = {
            "params": {
                "k": [1, 2, 4, 8],
                "k_group": [1, 2],
                "expertCount": [16],
                "group_count": [4],
            },
            "constraints": [
                {"formula": "k <= k_group * (expertCount / group_count)"},
                {"formula": "k_group < group_count"},
            ],
            "coverage": "high",
        }
        params = enumerate_params(spec)
        for p in params:
            pge = p["expertCount"] / p["group_count"]  # 4
            assert p["k"] <= p["k_group"] * pge, f"k constraint violated: {p}"
            assert p["k_group"] < p["group_count"], f"kg constraint violated: {p}"

    def test_modulo_formula(self):
        spec = {
            "params": {"expertCount": [64, 128, 256], "group_count": [3, 4, 8]},
            "constraints": [{"formula": "expertCount % group_count == 0"}],
            "coverage": "high",
        }
        params = enumerate_params(spec)
        for p in params:
            assert p["expertCount"] % p["group_count"] == 0, f"mod violated: {p}"
        # 64%3!=0, 128%3!=0, 256%3!=0 should all be filtered
        assert not any(p["group_count"] == 3 for p in params)

    def test_formula_skips_missing_vars(self):
        """If a variable in the formula is not in the case, skip (don't filter)."""
        spec = {
            "params": {"a": [1, 2]},
            "constraints": [{"formula": "a <= b"}],  # b not in params
            "coverage": "high",
        }
        params = enumerate_params(spec)
        assert len(params) == 2  # not filtered because b is missing

    def test_formula_with_if_then_combined(self):
        spec = {
            "params": {"k": [1, 2, 4], "expertCount": [4], "dst_type": [2, 40]},
            "constraints": [
                {"formula": "k <= expertCount"},
                {"if": {"dst_type": [2]}, "then": {"k": [1, 2]}},
            ],
            "coverage": "high",
        }
        params = enumerate_params(spec)
        for p in params:
            assert p["k"] <= p["expertCount"]
            if p["dst_type"] == 2:
                assert p["k"] in [1, 2]

    def test_division_by_zero_safe(self):
        spec = {
            "params": {"a": [10], "b": [0, 1]},
            "constraints": [{"formula": "a / b > 5"}],
            "coverage": "high",
        }
        params = enumerate_params(spec)
        # b=0 should not crash, just skip that constraint
        assert len(params) >= 1


class TestDescRules:
    """Tests for _desc generation from desc_rules."""

    def test_desc_from_formula_rules(self):
        spec = {
            "groups": [{
                "id": "g1",
                "desc_rules": [
                    {"formula": "batch < 64", "desc": "未开满核"},
                    {"formula": "batch >= 64", "desc": "开满核"},
                    {"formula": "x_last % 32 == 0", "desc": "对齐"},
                    {"formula": "x_last % 32 != 0", "desc": "非对齐"},
                ],
                "params": {"batch": [1, 64, 65], "x_last": [32, 30]},
            }],
            "coverage": "high",
        }
        params = enumerate_params(spec)
        p1 = next(p for p in params if p["batch"] == 1 and p["x_last"] == 30)
        assert p1["_desc"] == "未开满核; 非对齐"
        p2 = next(p for p in params if p["batch"] == 65 and p["x_last"] == 32)
        assert p2["_desc"] == "开满核; 对齐"

    def test_desc_from_low_config_note(self):
        spec = {
            "groups": [{
                "id": "g1",
                "low_configs": [{"x_last": 8192, "note": "LLaMA-7B"}],
                "params": {"x_last": [100]},
            }],
            "coverage": "low",
        }
        params = enumerate_params(spec)
        assert params[0]["_desc"] == "LLaMA-7B"

    def test_desc_from_if_rules(self):
        spec = {
            "groups": [{
                "id": "g1",
                "desc_rules": [
                    {"if": {"dst_type": [2]}, "desc": "int8输出"},
                    {"if": {"dst_type": [40, 41]}, "desc": "fp4输出"},
                ],
                "params": {"dst_type": [2, 40]},
            }],
            "coverage": "high",
        }
        params = enumerate_params(spec)
        assert next(p for p in params if p["dst_type"] == 2)["_desc"] == "int8输出"
        assert next(p for p in params if p["dst_type"] == 40)["_desc"] == "fp4输出"

    def test_no_desc_rules_no_desc(self):
        spec = {
            "groups": [{
                "id": "g1",
                "params": {"a": [1, 2]},
            }],
            "coverage": "high",
        }
        params = enumerate_params(spec)
        assert "_desc" not in params[0]
