#!/usr/bin/env python3
"""CLI entry point: validate param_def.json and run enumerator.

Usage:
    python run.py --param-def <path> --output_dir <dir> [--coverage medium] [--seed 42]
"""
import argparse
import json
import os
import sys

from engine.enumerator import enumerate_params, compute_coverage


def validate_spec(spec: dict, path: str) -> list[str]:
    """Validate param_def.json structure. Returns list of error messages."""
    errors = []

    if "params" not in spec and "groups" not in spec:
        errors.append("must have 'params' or 'groups' key")
        return errors

    if "groups" in spec:
        if not isinstance(spec["groups"], list):
            errors.append("'groups' must be a list")
            return errors

        for i, group in enumerate(spec["groups"]):
            prefix = f"groups[{i}]"
            if not isinstance(group, dict):
                errors.append(f"{prefix}: must be a dict")
                continue

            if "id" not in group:
                errors.append(f"{prefix}: missing 'id'")

            gid = group.get("id", f"group_{i}")
            prefix = f"group '{gid}'"

            # params check
            dims = group.get("params", {})
            low_configs = group.get("low_configs", [])
            if not dims and not low_configs:
                errors.append(f"{prefix}: must have 'params' or 'low_configs'")

            if not isinstance(dims, dict):
                errors.append(f"{prefix}: 'params' must be a dict")

            # params values check
            for dim_name, dim_def in dims.items():
                if isinstance(dim_def, list):
                    if len(dim_def) == 0:
                        errors.append(f"{prefix}.params.{dim_name}: empty list")
                elif isinstance(dim_def, dict):
                    if "min" not in dim_def and "thresholds" not in dim_def:
                        errors.append(f"{prefix}.params.{dim_name}: dict must have 'min' or 'thresholds'")
                else:
                    errors.append(f"{prefix}.params.{dim_name}: must be list or dict, got {type(dim_def).__name__}")

            # constraints check
            constraints = group.get("constraints", [])
            if not isinstance(constraints, list):
                errors.append(f"{prefix}: 'constraints' must be a list")
            else:
                for j, c in enumerate(constraints):
                    if isinstance(c, str):
                        continue  # text constraint, ok
                    if not isinstance(c, dict):
                        errors.append(f"{prefix}.constraints[{j}]: must be dict or string")
                        continue
                    valid_keys = {"if", "then", "requires", "formula", "desc"}
                    if not (c.keys() & valid_keys):
                        errors.append(f"{prefix}.constraints[{j}]: unrecognized format, needs 'if/then', 'requires', or 'formula'")

            # low_configs check
            if not isinstance(low_configs, list):
                errors.append(f"{prefix}: 'low_configs' must be a list")
            else:
                for j, cfg in enumerate(low_configs):
                    if not isinstance(cfg, dict):
                        errors.append(f"{prefix}.low_configs[{j}]: must be a dict")

            # desc_rules check
            desc_rules = group.get("desc_rules", [])
            if not isinstance(desc_rules, list):
                errors.append(f"{prefix}: 'desc_rules' must be a list")
            else:
                for j, rule in enumerate(desc_rules):
                    if not isinstance(rule, dict):
                        errors.append(f"{prefix}.desc_rules[{j}]: must be a dict")
                    elif "desc" not in rule:
                        errors.append(f"{prefix}.desc_rules[{j}]: missing 'desc' field")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Whitebox test parameter enumerator")
    parser.add_argument("--param-def", required=True, help="Path to param_def.json")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--coverage", default=None, choices=["low", "medium", "high"],
                        help="Override coverage level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load
    try:
        with open(args.param_def, encoding="utf-8") as f:
            spec = json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON syntax error in {args.param_def}: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"File not found: {args.param_def}", file=sys.stderr)
        sys.exit(1)

    # Validate
    errors = validate_spec(spec, args.param_def)
    if errors:
        print(f"Validation errors in {args.param_def}:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    # Enumerate
    params = enumerate_params(spec, seed=args.seed, coverage=args.coverage)

    # Write
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "cases.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    # Coverage report
    cov_report = compute_coverage(params, spec)
    cov_path = os.path.join(args.output_dir, "coverage_report.json")
    with open(cov_path, "w", encoding="utf-8") as f:
        json.dump(cov_report, f, indent=2, ensure_ascii=False, default=str)

    # Summary
    print(f"Generated {len(params)} parameter combinations")
    avg_pct = cov_report["summary"]["avg_pairwise_coverage_pct"]
    print(f"Pairwise coverage: {avg_pct}%")
    if any("_group" in p for p in params):
        from collections import Counter
        by_group = Counter(p.get("_group", "flat") for p in params)
        for g, cnt in sorted(by_group.items()):
            print(f"  {g}: {cnt}")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
