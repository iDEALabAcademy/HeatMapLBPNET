"""
Generic model size measurement utilities.

Counts learnable parameters only (requires_grad=True) and estimates bytes with
per-parameter bitwidth resolution. Supports mixed precision policies via a YAML
policy file and module/parameter attributes.

Outputs can be aggregated per module path, per module type, and per bitwidth.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _get_owner_module(model: nn.Module, param_full_name: str) -> Tuple[nn.Module, str]:
    """Return the owner module and the leaf param name for a given parameter path.

    Example: "backbone.stages.0.conv.weight" -> (module=..., leaf_name='weight').
    """
    parts = param_full_name.split(".")
    leaf = parts[-1]
    cur: nn.Module = model
    for p in parts[:-1]:
        if not hasattr(cur, p):
            # Handle ModuleList/Sequential numeric indexing
            try:
                idx = int(p)
                cur = cur[idx]  # type: ignore[index]
            except Exception:
                # Give up and return the last valid module
                break
        else:
            cur = getattr(cur, p)
    return cur, leaf


def _tensor_attr_bitwidth(t: torch.Tensor) -> Optional[int]:
    for key in ("_bitwidth", "bitwidth", "quant_bits"):
        if hasattr(t, key):
            try:
                return int(getattr(t, key))
            except Exception:
                continue
    return None


def _module_attr_bitwidth(m: nn.Module) -> Optional[int]:
    for key in ("w_bit", "weight_bit", "bitwidth", "n_bits", "n_bits_per_out"):
        if hasattr(m, key):
            try:
                return int(getattr(m, key))
            except Exception:
                continue
    return None


@dataclass
class PolicyRule:
    name_regex: Optional[re.Pattern] = None
    module_type: Optional[re.Pattern] = None
    bitwidth: Optional[int] = None
    include: Optional[bool] = None
    prefer_module_attr: Optional[List[str]] = None


@dataclass
class Policy:
    default_bitwidth: int = 32
    default_include: bool = True
    rules: List[PolicyRule] = None  # type: ignore[assignment]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Policy":
        defaults = d.get("defaults", {}) or {}
        rules_in = d.get("rules", []) or []
        rules: List[PolicyRule] = []
        for r in rules_in:
            name_re = r.get("name_regex")
            mod_type = r.get("module_type")
            compiled_name = re.compile(name_re) if isinstance(name_re, str) else None
            compiled_type = re.compile(mod_type) if isinstance(mod_type, str) else None
            rules.append(
                PolicyRule(
                    name_regex=compiled_name,
                    module_type=compiled_type,
                    bitwidth=r.get("bitwidth"),
                    include=r.get("include"),
                    prefer_module_attr=r.get("prefer_module_attr"),
                )
            )
        return Policy(
            default_bitwidth=_safe_int(defaults.get("bitwidth", 32), 32),
            default_include=bool(defaults.get("include", True)),
            rules=rules,
        )


def load_policy(path: Optional[str]) -> Policy:
    if not path:
        return Policy()
    try:
        import yaml  # type: ignore
    except Exception:
        # YAML not available; fallback to defaults
        return Policy()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Policy.from_dict(data or {})


def _apply_policy(name: str, module_type: str, policy: Policy) -> Tuple[Optional[int], Optional[bool], Optional[List[str]]]:
    # Priority: name_regex, then module_type
    for rule in policy.rules or []:
        if rule.name_regex and rule.name_regex.search(name):
            return rule.bitwidth, rule.include, rule.prefer_module_attr
    for rule in policy.rules or []:
        if rule.module_type and rule.module_type.search(module_type):
            return rule.bitwidth, rule.include, rule.prefer_module_attr
    return None, None, None


def _builtin_bitwidth(name: str, module: nn.Module, module_type: str) -> Optional[int]:
    lname = name.lower()
    mname = module_type.lower()
    # RP-related: prefer module attr n_bits_per_out
    if ("rp" in lname) or ("rp" in mname):
        mbw = _module_attr_bitwidth(module)
        if mbw is not None:
            return mbw
        # fallback heuristic
        return 4
    # LBP scalar params
    if any(k in lname for k in ("alpha", "offset", "threshold")):
        return 16
    # First conv / last linear heuristics
    if lname.endswith("weight") and isinstance(module, nn.Conv2d):
        # crude heuristic: stem or very early conv
        if ("stem" in name) or re.search(r"(^|\.)conv(1|0)\.", name):
            return 8
    if lname.endswith("weight") and isinstance(module, nn.Linear):
        # crude heuristic for classifier
        if any(x in name for x in ("classifier", "fc", "head", "fc_layers")):
            return 16
    return None


def _decide_bitwidth(
    param: torch.Tensor,
    module: nn.Module,
    name: str,
    policy: Policy,
    prefer_module_attrs: Optional[List[str]] = None,
) -> Tuple[int, bool, str]:
    # 1) Parameter-level explicit attribute
    bw = _tensor_attr_bitwidth(param)
    if bw is not None:
        return _safe_int(bw, policy.default_bitwidth), True, "param_attr"

    # 2) Module-level attrs
    if prefer_module_attrs:
        for key in prefer_module_attrs:
            if hasattr(module, key):
                try:
                    bw = int(getattr(module, key))
                    return _safe_int(bw, policy.default_bitwidth), True, f"module_attr:{key}"
                except Exception:
                    pass
    mbw = _module_attr_bitwidth(module)
    if mbw is not None:
        return _safe_int(mbw, policy.default_bitwidth), True, "module_attr"

    # 3) Built-in rules
    bbw = _builtin_bitwidth(name, module, module.__class__.__name__)
    if bbw is not None:
        return _safe_int(bbw, policy.default_bitwidth), True, "builtin"

    # 4) Policy defaults
    return policy.default_bitwidth, policy.default_include, "default"


def _bytes_for(numel: int, bitwidth: int) -> int:
    return (int(numel) * int(bitwidth) + 7) // 8


def measure_model_size(
    model: nn.Module,
    policy: Optional[Policy] = None,
    strict: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    policy = policy or Policy()

    rows: List[Dict[str, Any]] = []
    by_type: Dict[str, int] = {}
    by_bit: Dict[int, Dict[str, int]] = {}
    weight_only_total = 0
    full_total = 0

    weight_name_whitelist = ("weight", "kernel", "proj", "weight_q")

    for full_name, p in model.named_parameters(recurse=True):
        if not p.requires_grad:
            continue
        owner, leaf = _get_owner_module(model, full_name)
        mtype = owner.__class__.__name__

        # Policy matching first to fetch include/bit hints
        pol_bw, pol_inc, prefer_keys = _apply_policy(full_name, mtype, policy)

        # Decide bitwidth
        if pol_bw is not None:
            bitwidth, src = _safe_int(pol_bw, policy.default_bitwidth), "policy"
        else:
            bitwidth, _, src = _decide_bitwidth(p, owner, full_name, policy, prefer_keys)

        include = policy.default_include if pol_inc is None else bool(pol_inc)

        if strict and (bitwidth is None):
            raise RuntimeError(f"Strict mode: missing bitwidth for {full_name}")

        if not include:
            rows.append({
                "name": full_name,
                "module_type": mtype,
                "shape": tuple(p.shape),
                "numel": p.numel(),
                "bitwidth": bitwidth,
                "bytes": 0,
                "source": src,
                "group": "excluded",
            })
            continue

        numel = int(p.numel())
        bytes_ = 0 if dry_run else _bytes_for(numel, bitwidth)

        is_weighty = any(full_name.endswith(w) for w in weight_name_whitelist)
        if is_weighty:
            weight_only_total += bytes_
        full_total += bytes_

        rows.append({
            "name": full_name,
            "module_type": mtype,
            "shape": tuple(p.shape),
            "numel": numel,
            "bitwidth": bitwidth,
            "bytes": bytes_,
            "source": src,
            "group": "weight_only" if is_weighty else "full",
        })

        # Aggregations
        by_type[mtype] = by_type.get(mtype, 0) + bytes_
        if bitwidth not in by_bit:
            by_bit[bitwidth] = {"params": 0, "bytes": 0}
        by_bit[bitwidth]["params"] += numel
        by_bit[bitwidth]["bytes"] += bytes_

    result = {
        "weight_only_total_bytes": weight_only_total,
        "full_params_total_bytes": full_total,
        "rows": rows,
        "by_type": by_type,
        "by_bitwidth": by_bit,
    }
    return result


def to_human_readable(bytes_val: int) -> Tuple[float, float]:
    kb = float(bytes_val) / 1024.0
    mb = kb / 1024.0
    return kb, mb


def save_reports(result: Dict[str, Any], out_dir: str) -> Tuple[str, str]:
    import csv
    import os
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "size_report.csv")
    json_path = os.path.join(out_dir, "size_report.json")

    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "module_type", "shape", "numel", "bitwidth", "bytes", "source", "group"])
        for r in result.get("rows", []):
            writer.writerow([
                r["name"], r["module_type"], str(tuple(r["shape"])), r["numel"], r["bitwidth"], r["bytes"], r["source"], r["group"]
            ])

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return csv_path, json_path










