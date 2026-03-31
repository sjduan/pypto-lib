"""Parameter enumerator: threshold expansion + pairwise/cartesian combination."""

from __future__ import annotations

import itertools
import json
import random
from typing import Any


# ---------------------------------------------------------------------------
# Threshold expansion
# ---------------------------------------------------------------------------

def _expand_thresholds(thresholds: list[dict]) -> list[int]:
    """Expand threshold definitions into boundary values."""
    values: list[int] = []
    for th in thresholds:
        t = th["type"]
        v = th["value"]
        if t == "branch_split":
            values.extend([v - 1, v, v + 1])
            for m in th.get("multiples", []):
                values.append(m * v)
        elif t == "alignment":
            for k in range(1, 4):
                values.extend([k * v - 1, k * v, k * v + 1])
        elif t == "divisor":
            for k in range(1, 4):
                values.extend([k * v, k * v + 1])
        else:
            raise ValueError(f"Unknown threshold type: {t}")
    return values


def expand_dimension(dim_def: Any, seed: int = 42) -> list:
    """Expand a dimension definition into concrete values.

    *dim_def* is either a plain list (returned as-is) or a dict with
    ``thresholds``, ``min``, ``max``, and optional ``alignment`` /
    ``random_count`` fields.
    """
    if isinstance(dim_def, list):
        return dim_def

    thresholds = dim_def.get("thresholds", [])
    lo = dim_def.get("min", 1)
    hi = dim_def.get("max", 2**31)
    align = dim_def.get("alignment")
    random_count = dim_def.get("random_count", 0)

    values = _expand_thresholds(thresholds)

    # Filter by min/max
    values = [v for v in values if lo <= v <= hi]

    # Filter by alignment
    if align:
        values = [v for v in values if v % align == 0]

    # Inject min and source_max as guaranteed boundary values
    # (respecting alignment if set)
    source_max = dim_def.get("source_max")
    if lo not in values and (not align or lo % align == 0):
        values.append(lo)
    if source_max is not None and source_max not in values and lo <= source_max <= hi:
        if not align or source_max % align == 0:
            values.append(source_max)

    # Deduplicate while preserving order
    seen: set[int] = set()
    unique: list[int] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    values = unique

    # Add random interior values (without materializing full range)
    if random_count > 0:
        rng = random.Random(seed)
        generated = set()
        attempts = 0
        max_attempts = random_count * 20
        while len(generated) < random_count and attempts < max_attempts:
            if align:
                # Generate aligned random value directly
                lo_aligned = lo + (align - lo % align) % align
                hi_aligned = hi - hi % align
                if lo_aligned <= hi_aligned:
                    k = rng.randint(lo_aligned // align, hi_aligned // align)
                    v = k * align
                else:
                    break
            else:
                v = rng.randint(lo, hi)
            if v not in seen and v not in generated:
                generated.add(v)
            attempts += 1
        values.extend(sorted(generated))

    return values


# ---------------------------------------------------------------------------
# Combination strategies
# ---------------------------------------------------------------------------

def _make_hashable(v: Any) -> Any:
    """Make a value hashable for set operations."""
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True)
    return v


def _one_at_a_time(dim_names: list[str], dim_values: list[list]) -> list[dict]:
    """Low coverage: vary each dimension independently, others at first value."""
    if not dim_names:
        return [{}]

    defaults = {name: vals[0] for name, vals in zip(dim_names, dim_values)}
    seen_tuples: set[tuple] = set()
    cases: list[dict] = []

    for i, name in enumerate(dim_names):
        for val in dim_values[i]:
            row = dict(defaults)
            row[name] = val
            key = tuple(row[n] if not isinstance(row[n], (dict, list))
                        else json.dumps(row[n], sort_keys=True)
                        for n in dim_names)
            if key not in seen_tuples:
                seen_tuples.add(key)
                cases.append(row)

    return cases


def _pairwise_ipo(dim_names: list[str], dim_values: list[list],
                   seed: int = 42) -> list[dict]:
    """Medium coverage: randomized In-Parameter-Order pairwise algorithm.

    Improvements over classic IPO:
    - (A) Tie-breaking uses random selection instead of always picking first value
    - (B) Dimension order is randomized (different dims get full cross-product seed)
    - (C) Post-pairwise balance pass adds cases for under-represented values
    """
    if not dim_names:
        return [{}]
    if len(dim_names) == 1:
        return [{dim_names[0]: v} for v in dim_values[0]]

    rng = random.Random(seed)

    # (B) Randomize dimension order for seed phase diversity
    indices = list(range(len(dim_names)))
    rng.shuffle(indices)
    shuffled_names = [dim_names[i] for i in indices]
    shuffled_values = [dim_values[i] for i in indices]

    # Seed: full cross of first 2 (randomized) params
    rows: list[list] = []
    for a, b in itertools.product(shuffled_values[0], shuffled_values[1]):
        rows.append([a, b])

    for dim_idx in range(2, len(shuffled_names)):
        new_vals = shuffled_values[dim_idx]

        # Collect all uncovered pairs between new dim and each prior dim
        uncovered: set[tuple] = set()
        for prior in range(dim_idx):
            for pv in shuffled_values[prior]:
                for nv in new_vals:
                    uncovered.add((prior, _make_hashable(pv), _make_hashable(nv)))

        def _remove_covered(row: list, uncov: set[tuple]):
            nv_h = _make_hashable(row[dim_idx]) if dim_idx < len(row) else None
            if nv_h is None:
                return
            for prior in range(dim_idx):
                key = (prior, _make_hashable(row[prior]), nv_h)
                uncov.discard(key)

        # Horizontal extension: add best new-dim value to each existing row
        # (A) Tie-breaking: collect all best candidates, pick randomly
        for row in rows:
            best_count = -1
            candidates = []
            for nv in new_vals:
                count = 0
                nv_h = _make_hashable(nv)
                for prior in range(dim_idx):
                    key = (prior, _make_hashable(row[prior]), nv_h)
                    if key in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    candidates = [nv]
                elif count == best_count:
                    candidates.append(nv)
            row.append(rng.choice(candidates))
            _remove_covered(row, uncovered)

        # Vertical extension: add new rows for remaining uncovered pairs
        while uncovered:
            prior_idx, pv_h, nv_h = next(iter(uncovered))
            new_row = [None] * (dim_idx + 1)
            for v in shuffled_values[prior_idx]:
                if _make_hashable(v) == pv_h:
                    new_row[prior_idx] = v
                    break
            for v in new_vals:
                if _make_hashable(v) == nv_h:
                    new_row[dim_idx] = v
                    break
            # Fill remaining: collect tied candidates, pick randomly
            for col in range(dim_idx + 1):
                if new_row[col] is not None:
                    continue
                best_count = -1
                candidates = []
                for cv in shuffled_values[col]:
                    count = 0
                    cv_h = _make_hashable(cv)
                    key_new = (col, cv_h, _make_hashable(new_row[dim_idx]))
                    if key_new in uncovered:
                        count += 1
                    if count > best_count:
                        best_count = count
                        candidates = [cv]
                    elif count == best_count:
                        candidates.append(cv)
                new_row[col] = rng.choice(candidates)
            rows.append(new_row)
            _remove_covered(new_row, uncovered)

    # Un-shuffle: map back to original dimension order
    reverse_map = [0] * len(indices)
    for new_pos, orig_pos in enumerate(indices):
        reverse_map[orig_pos] = new_pos
    cases = [{dim_names[i]: row[reverse_map[i]] for i in range(len(dim_names))} for row in rows]

    # (C) Balance pass: ensure each value of each dimension appears at least
    #     ceil(total / num_values) * 0.5 times (minimum representation)
    from collections import Counter
    total = len(cases)
    for di, name in enumerate(dim_names):
        vals = dim_values[di]
        if len(vals) <= 1:
            continue
        counts = Counter(c[name] for c in cases)
        min_target = max(1, total // (len(vals) * 2))
        for v in vals:
            deficit = min_target - counts.get(v, 0)
            for _ in range(deficit):
                # Create a new case with this value, other dims random
                new_case = {name: v}
                for dj, other_name in enumerate(dim_names):
                    if dj != di:
                        new_case[other_name] = rng.choice(dim_values[dj])
                cases.append(new_case)

    return cases


def _cartesian(dim_names: list[str], dim_values: list[list]) -> list[dict]:
    """High coverage: full cartesian product."""
    if not dim_names:
        return [{}]
    combos = itertools.product(*dim_values)
    return [{dim_names[i]: vals[i] for i in range(len(dim_names))} for vals in combos]


# ---------------------------------------------------------------------------
# Constraint filtering
# ---------------------------------------------------------------------------

def _eval_formula(formula: str, case: dict) -> bool:
    """Evaluate a formula constraint against a case dict.

    Supports arithmetic (+, -, *, /, //, %), comparisons (<=, >=, <, >, ==, !=),
    and logical (and, or, not). Variable names are looked up in the case dict.
    Returns True if the formula holds, False if it doesn't.
    Skips (returns True) if any referenced variable is missing from the case.
    """
    # Build a safe namespace from case values (only numeric)
    ns = {}
    for k, v in case.items():
        if k.startswith("_"):
            continue
        if isinstance(v, (int, float)):
            ns[k] = v
    # Check if all variables in formula are available
    import re
    var_names = set(re.findall(r'\b([a-zA-Z_]\w*)\b', formula))
    keywords = {'and', 'or', 'not', 'True', 'False', 'in'}
    needed = var_names - keywords - set(dir(__builtins__) if isinstance(__builtins__, dict) else dir(__builtins__))
    for name in needed:
        if name not in ns:
            return True  # skip if variable not in case
    try:
        return bool(eval(formula, {"__builtins__": {}}, ns))
    except (ZeroDivisionError, TypeError, NameError):
        return True  # skip on eval errors


def _apply_constraints(cases: list[dict], constraints: list) -> list[dict]:
    """Filter cases by constraint rules.

    Supported constraint formats::

        {"if": {"dst_type": [2, 35, 36]}, "then": {"round_mode": ["rint"]}}
        {"requires": {"x_dtype": "int32", "quant_mode": "static"}}
        {"formula": "k <= k_group * (expertCount / group_count)"}

    An ``if/then`` constraint means: when *all* ``if`` conditions match,
    *at least one* ``then`` condition must also match.  A ``requires``
    constraint means: when any key in the dict is present in the case,
    all values must match exactly.  A ``formula`` constraint is a Python
    expression evaluated against the case's numeric fields.
    """
    if not constraints:
        return cases

    def _matches_condition(case: dict, cond: dict) -> bool:
        for key, expected in cond.items():
            val = case.get(key)
            if val is None:
                continue
            if isinstance(expected, list):
                if val not in expected:
                    return False
            else:
                if val != expected:
                    return False
        return True

    def _passes(case: dict) -> bool:
        for c in constraints:
            if not isinstance(c, dict):
                continue  # skip text constraints

            if "formula" in c:
                if not _eval_formula(c["formula"], case):
                    return False

            elif "if" in c and "then" in c:
                if _matches_condition(case, c["if"]):
                    if not _matches_condition(case, c["then"]):
                        return False

            elif "requires" in c:
                req = c["requires"]
                relevant = any(k in case for k in req)
                if relevant and not _matches_condition(case, req):
                    return False

        return True

    return [c for c in cases if _passes(c)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _expand_and_combine(
    dims: dict, cov: str, seed: int,
    group_id: str | None = None,
    constraints: list | None = None,
) -> list[dict]:
    """Expand params, combine into parameter dicts, apply constraints."""
    dim_names: list[str] = list(dims.keys())
    dim_values: list[list] = [expand_dimension(dims[n], seed=seed) for n in dim_names]

    if cov == "low":
        cases = _one_at_a_time(dim_names, dim_values)
    elif cov == "medium":
        cases = _pairwise_ipo(dim_names, dim_values, seed=seed)
    elif cov == "high":
        cases = _cartesian(dim_names, dim_values)
    else:
        raise ValueError(f"Unknown coverage level: {cov}")

    if group_id is not None:
        for c in cases:
            c["_group"] = group_id

    # Apply constraint filtering
    if constraints:
        cases = _apply_constraints(cases, constraints)

    return cases


def _build_desc(case: dict, desc_rules: list) -> str | None:
    """Build a description string for a case by matching desc_rules.

    Each rule: {"formula": "expr", "desc": "text"} or {"if": {...}, "desc": "text"}.
    All matching descriptions are joined with "; ".
    For low_configs, the "note" field is used directly.
    """
    if "note" in case:
        return case["note"]

    if not desc_rules:
        return None

    parts = []
    for rule in desc_rules:
        if not isinstance(rule, dict) or "desc" not in rule:
            continue
        if "formula" in rule:
            if _eval_formula(rule["formula"], case):
                parts.append(rule["desc"])
        elif "if" in rule:
            matched = True
            for k, expected in rule["if"].items():
                val = case.get(k)
                if isinstance(expected, list):
                    if val not in expected:
                        matched = False
                        break
                elif val != expected:
                    matched = False
                    break
            if matched:
                parts.append(rule["desc"])

    return "; ".join(parts) if parts else None


def enumerate_params(
    spec: dict,
    seed: int = 42,
    coverage: str | None = None,
) -> list[dict]:
    """Generate parameter combinations from a param_def spec.

    Supports two formats:

    **Flat** (single dimension set)::

        {"params": {"name": values_or_def, ...}, "coverage": "medium"}

    **Grouped** (per-group dimension sets, each group is a test focus area)::

        {"groups": [
            {"id": "group_a", "params": {...}, "constraints": [...]},
            {"id": "group_b", "params": {...}},
        ], "coverage": "medium"}

    Each group is expanded independently, constraints are applied per-group,
    then results are merged.
    A ``_group`` field is added to each result dict to identify its source group.

    Constraint formats supported in the ``constraints`` list::

        {"if": {"dst_type": [2, 35]}, "then": {"round_mode": ["rint"]}}
        {"requires": {"x_dtype": "int32", "quant_mode": "static"}}

    Text-only constraints (plain strings) are preserved for documentation
    but skipped during filtering.
    """
    cov = coverage or spec.get("coverage", "low")

    # Grouped format
    if "groups" in spec:
        all_cases: list[dict] = []
        for group in spec["groups"]:
            gid = group.get("id", f"group_{len(all_cases)}")
            network_configs = group.get("low_configs", [])

            # low coverage: only network common shapes
            if cov == "low" and network_configs:
                desc_rules = group.get("desc_rules", [])
                for cfg in network_configs:
                    case = dict(cfg)
                    case["_group"] = gid
                    desc = _build_desc(case, desc_rules)
                    if desc:
                        case["_desc"] = desc
                    all_cases.append(case)
                continue

            # medium/high: combinatorial expansion + network common shapes
            dims = group.get("params", {})
            constraints = group.get("constraints", [])
            if not dims and not network_configs:
                continue
            cases = []
            if dims:
                cases = _expand_and_combine(dims, cov, seed, group_id=gid,
                                             constraints=constraints)
            # Append network configs (dedup by content)
            if network_configs:
                existing = {json.dumps(c, sort_keys=True, default=str) for c in cases}
                for cfg in network_configs:
                    case = dict(cfg)
                    case["_group"] = gid
                    key = json.dumps(case, sort_keys=True, default=str)
                    if key not in existing:
                        cases.append(case)
                        existing.add(key)

            # Add descriptions
            desc_rules = group.get("desc_rules", [])
            if desc_rules or any("note" in c for c in cases):
                for case in cases:
                    desc = _build_desc(case, desc_rules)
                    if desc:
                        case["_desc"] = desc

            all_cases.extend(cases)
        return all_cases

    # Flat format (backward compatible)
    dims = spec.get("params", {})
    constraints = spec.get("constraints", [])
    if not dims:
        return []
    return _expand_and_combine(dims, cov, seed, constraints=constraints)


def compute_coverage(cases: list[dict], spec: dict) -> dict:
    """Compute single-factor and pairwise coverage metrics.

    Returns a dict with:
      - single_factor: per-dimension coverage (which values appear)
      - pairwise: per-dimension-pair coverage (which value pairs appear)
      - missing_pairs: list of uncovered (dim_a, val_a, dim_b, val_b) tuples
    """
    # Collect dimension names and their expected values per group
    groups = spec.get("groups", [])
    if not groups and "params" in spec:
        groups = [{"id": "flat", "params": spec["params"]}]

    report: dict = {"groups": {}}

    for group in groups:
        gid = group.get("id", "flat")
        dims = group.get("params", {})
        if not dims:
            continue

        # Get cases for this group
        group_cases = [c for c in cases if c.get("_group", "flat") == gid]
        if not group_cases:
            continue

        dim_names = [d for d in dims if not d.startswith("_")]

        # Collect actual values per dimension from cases
        actual_values: dict[str, set] = {d: set() for d in dim_names}
        for case in group_cases:
            for d in dim_names:
                if d in case:
                    actual_values[d].add(case[d])

        # Expected values per dimension (expand if needed)
        expected_values: dict[str, set] = {}
        for d in dim_names:
            v = dims[d]
            if isinstance(v, list):
                expected_values[d] = set(v)
            else:
                # Threshold-based: use actual values from cases as expected
                expected_values[d] = actual_values[d]

        # Single-factor coverage
        single: dict = {}
        for d in dim_names:
            expected = expected_values[d]
            covered = actual_values[d] & expected
            single[d] = {
                "expected": len(expected),
                "covered": len(covered),
                "missing": sorted(str(v) for v in expected - covered),
            }

        # Pairwise coverage
        pairwise: dict = {}
        missing_pairs: list = []
        for i, d1 in enumerate(dim_names):
            for d2 in dim_names[i + 1:]:
                pair_key = f"{d1} × {d2}"
                # Expected pairs
                expected_pairs = set(
                    itertools.product(expected_values[d1], expected_values[d2])
                )
                # Actual pairs
                actual_pairs: set = set()
                for case in group_cases:
                    v1 = case.get(d1)
                    v2 = case.get(d2)
                    if v1 is not None and v2 is not None:
                        actual_pairs.add((v1, v2))
                covered_pairs = actual_pairs & expected_pairs
                uncovered = expected_pairs - actual_pairs
                pct = len(covered_pairs) / len(expected_pairs) * 100 if expected_pairs else 100
                pairwise[pair_key] = {
                    "expected": len(expected_pairs),
                    "covered": len(covered_pairs),
                    "coverage_pct": round(pct, 1),
                }
                if uncovered and len(uncovered) <= 20:
                    for v1, v2 in sorted(uncovered, key=str):
                        missing_pairs.append({
                            "group": gid, "dim1": d1, "val1": v1,
                            "dim2": d2, "val2": v2,
                        })

        report["groups"][gid] = {
            "case_count": len(group_cases),
            "single_factor": single,
            "pairwise": pairwise,
            "missing_pairs_sample": missing_pairs[:50],
        }

    # Overall stats
    total_cases = len(cases)
    all_pairwise = [
        p for g in report["groups"].values()
        for p in g["pairwise"].values()
    ]
    avg_pct = (
        sum(p["coverage_pct"] for p in all_pairwise) / len(all_pairwise)
        if all_pairwise else 100
    )
    report["summary"] = {
        "total_cases": total_cases,
        "avg_pairwise_coverage_pct": round(avg_pct, 1),
    }
    return report


def enumerate_params_from_file(
    path: str,
    coverage: str | None = None,
    seed: int = 42,
) -> list[dict]:
    """Load a JSON spec file and enumerate parameters."""
    with open(path) as f:
        spec = json.load(f)
    return enumerate_params(spec, seed=seed, coverage=coverage)
