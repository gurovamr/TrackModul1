#!/usr/bin/env python3
"""
Patient-specific Circle of Willis geometry injection for first_blood (Abel_ref2 template).

Run from:
  <repo_root>/pipeline

Reads patient raw data from:
  <repo_root>/data/
    data/cow_features/{topcow_<modality>_<pid>.json | feature_<modality>_<pid>.json}
    data/cow_nodes/{topcow_<modality>_<pid>.json | nodes_<modality>_<pid>.json}
    data/cow_variants/{topcow_<modality>_<pid>.json | variant_<modality>_<pid>.json}  (optional)

Creates:
  <repo_root>/models/patient_<pid>/
    - copies ALL template CSVs except arterial.csv
    - writes generated arterial.csv
    - modifications_log.csv
    - missing_mapping_log.csv
    - skipped_segments_log.csv
    - unmapped_patient_keys_log.csv
    - variant_ignored_keys_log.csv
    - variant_conflicts_log.csv

Design:
- No invented wall thickness: thickness columns are NOT overwritten.
- Side inference priority:
    (1) explicit "R-" / "L-" hints from nodes JSON keys
    (2) x-coordinate fallback
- "Absent vessel policy" (BC-consistent):
    * Only apply absence from variants["anterior"] and variants["posterior"] where value == False
    * Ignore variants["fetal"] and variants["fenestration"] for absence decisions
    * Never occlude if we measured that vessel ("measured beats variants")
- Basilar (BA) split across A59 and A56 proportional to template lengths.

Important:
- We DO NOT map ACA->A1 or PCA->P1. Only map explicit canonical segments:
    ICA, MCA, A1, A2, P1, P2, Pcom, Acom, BA.
  This prevents accidental overwrites when both PCA and P1 exist in the raw data.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


# -------------------------
# Repo-relative paths
# -------------------------

def get_repo_root() -> Path:
    pipeline_dir = Path(__file__).resolve().parent
    return pipeline_dir.parent


# -------------------------
# Utilities
# -------------------------

def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _walk_json(obj, on_dict=None, on_list=None, path=()):
    if isinstance(obj, dict):
        if on_dict:
            on_dict(obj, path)
        for k, v in obj.items():
            _walk_json(v, on_dict=on_dict, on_list=on_list, path=path + (str(k),))
    elif isinstance(obj, list):
        if on_list:
            on_list(obj, path)
        for i, v in enumerate(obj):
            _walk_json(v, on_dict=on_dict, on_list=on_list, path=path + (str(i),))


# -------------------------
# File discovery
# -------------------------

def find_patient_files(data_root: Path, pid: str) -> Tuple[str, Path, Path, Optional[Path]]:
    """
    Supports both:
      topcow_<modality>_<pid>.json
      feature_<modality>_<pid>.json / nodes_<modality>_<pid>.json / variant_<modality>_<pid>.json
    Preference: ct then mr
    """
    pid = str(pid).zfill(3)

    for modality in ["ct", "mr"]:
        # topcow naming
        feat_a = data_root / "cow_features" / f"topcow_{modality}_{pid}.json"
        nods_a = data_root / "cow_nodes" / f"topcow_{modality}_{pid}.json"
        var_a  = data_root / "cow_variants" / f"topcow_{modality}_{pid}.json"

        # legacy naming
        feat_b = data_root / "cow_features" / f"feature_{modality}_{pid}.json"
        nods_b = data_root / "cow_nodes" / f"nodes_{modality}_{pid}.json"
        var_b  = data_root / "cow_variants" / f"variant_{modality}_{pid}.json"

        if feat_a.exists() and nods_a.exists():
            return modality, feat_a, nods_a, (var_a if var_a.exists() else None)
        if feat_b.exists() and nods_b.exists():
            return modality, feat_b, nods_b, (var_b if var_b.exists() else None)

    raise FileNotFoundError(
        f"Could not find patient files for pid={pid} under {data_root}.\n"
        f"Expected one of:\n"
        f"  {data_root}/cow_features/topcow_[ct|mr]_{pid}.json\n"
        f"  {data_root}/cow_nodes/topcow_[ct|mr]_{pid}.json\n"
        f"  {data_root}/cow_variants/topcow_[ct|mr]_{pid}.json (optional)\n"
        f"OR:\n"
        f"  {data_root}/cow_features/feature_[ct|mr]_{pid}.json\n"
        f"  {data_root}/cow_nodes/nodes_[ct|mr]_{pid}.json\n"
        f"  {data_root}/cow_variants/variant_[ct|mr]_{pid}.json (optional)\n"
    )


# -------------------------
# Nodes + side inference (STRONG)
# -------------------------

def flatten_nodes(nodes_json: Any):
    """
    Collect dicts like {"id": ..., "coords": [x,y,z]}.
    Also create node_id -> side_hint from ANY ancestor key containing 'R-' or 'L-'.

    Returns:
      id_to_xyz: dict[int] -> np.array([x,y,z])
      node_side_hint: dict[int] -> 'R'|'L'
      r_x, l_x: samples of x coords from explicit R/L hints (for x-sign fallback orientation)
    """
    id_to_xyz: Dict[int, np.ndarray] = {}
    node_side_hint: Dict[int, str] = {}
    r_x: List[float] = []
    l_x: List[float] = []

    def path_has_side(path) -> Optional[str]:
        # Strongest: any key containing "R-" or "L-" (not just startswith)
        for p in reversed(path):
            if not isinstance(p, str):
                continue
            if "R-" in p:
                return "R"
            if "L-" in p:
                return "L"
        return None

    def on_dict(d, path):
        if "id" in d and "coords" in d:
            try:
                nid = int(d["id"])
                coords = d["coords"]
                if isinstance(coords, (list, tuple)) and len(coords) >= 3:
                    xyz = np.array([float(coords[0]), float(coords[1]), float(coords[2])], dtype=float)
                    id_to_xyz[nid] = xyz

                    sh = path_has_side(path)
                    if sh in ("R", "L"):
                        # If conflicting hints appear for the same id, drop hint (safer)
                        if nid in node_side_hint and node_side_hint[nid] != sh:
                            node_side_hint.pop(nid, None)
                        else:
                            node_side_hint[nid] = sh
                            if sh == "R":
                                r_x.append(float(xyz[0]))
                            else:
                                l_x.append(float(xyz[0]))
            except Exception:
                return

    _walk_json(nodes_json, on_dict=on_dict)
    return id_to_xyz, node_side_hint, r_x, l_x


def infer_right_is_positive_x(r_x: List[float], l_x: List[float]) -> bool:
    # If we have explicit labels, infer orientation from their means (even if both negative)
    if len(r_x) > 0 and len(l_x) > 0:
        return float(np.mean(r_x)) > float(np.mean(l_x))
    # Otherwise assume Right = +x
    return True


def side_from_endpoints(
    id_to_xyz: Dict[int, np.ndarray],
    node_side_hint: Dict[int, str],
    start_id: int,
    end_id: int,
    right_is_positive_x: bool
) -> Optional[str]:
    """
    Priority:
      1) explicit node_side_hint if available on endpoints
      2) x-sign fallback using right_is_positive_x
    """
    # 1) explicit hint
    hints = []
    if start_id in node_side_hint:
        hints.append(node_side_hint[start_id])
    if end_id in node_side_hint:
        hints.append(node_side_hint[end_id])

    if hints:
        if all(h == hints[0] for h in hints):
            return hints[0]
        # conflicting explicit hints -> unknown
        return None

    # 2) fallback to x sign
    xs = []
    if start_id in id_to_xyz:
        xs.append(float(id_to_xyz[start_id][0]))
    if end_id in id_to_xyz:
        xs.append(float(id_to_xyz[end_id][0]))
    if not xs:
        return None
    x_mean = float(np.mean(xs))
    if right_is_positive_x:
        return "R" if x_mean >= 0.0 else "L"
    return "R" if x_mean <= 0.0 else "L"


# -------------------------
# Feature parsing
# -------------------------

def parse_patient_features(features_json: Any) -> List[Dict[str, Any]]:
    """
    Find dict entries containing:
      - segment: {start, end}
      - length (mm)
      - radius: {mean} (mm)

    Uses last non-numeric key from JSON path as raw_name (e.g. "P1", "A2", "Pcom"...).
    """
    segs: List[Dict[str, Any]] = []

    def on_dict(d, path):
        if "segment" not in d or "length" not in d or "radius" not in d:
            return
        seg = d.get("segment", {})
        rad = d.get("radius", {})
        if not isinstance(seg, dict) or not isinstance(rad, dict):
            return
        if "start" not in seg or "end" not in seg or "mean" not in rad:
            return

        try:
            start_id = int(seg["start"])
            end_id = int(seg["end"])
            length_mm = float(d["length"])
            radius_mm = float(rad["mean"])
        except Exception:
            return

        raw_name = None
        for p in reversed(path):
            if not str(p).isdigit():
                raw_name = str(p)
                break
        if raw_name is None:
            raw_name = "UNKNOWN"

        if "bifurcation" in raw_name.lower():
            return

        segs.append(
            {
                "raw_name": raw_name.strip(),
                "start_id": start_id,
                "end_id": end_id,
                "length_mm": length_mm,
                "radius_mm": radius_mm,
            }
        )

    _walk_json(features_json, on_dict=on_dict)
    return segs


# -------------------------
# Name normalization (NO SYNONYMS)
# -------------------------

CANONICAL = {"A1", "A2", "Acom", "Pcom", "MCA", "BA", "P1", "P2", "ICA"}  # strict

def normalize_segment_name(raw_name: str) -> str:
    """
    Strict normalization:
    - Keep exact canonical tokens only.
    - Minimal case/spacing normalization for Acom/Pcom/BA.
    - Everything else is returned as-is (and will be unmapped/ignored).
    """
    name = str(raw_name).strip()

    if name in CANONICAL:
        return name

    low = name.lower().replace(" ", "")
    if low == "acom":
        return "Acom"
    if low == "pcom":
        return "Pcom"
    if low == "ba":
        return "BA"

    # Do NOT map PCA->P1 or ACA->A1 etc. (no synonym mapping)
    return name


# -------------------------
# FirstBlood mapping (STRICT, NO SYNONYMS)
# -------------------------

def build_firstblood_mapping() -> Dict[Tuple[Optional[str], str], List[str]]:
    """
    Abel_ref2 IDs for the segments we support.

    NOTE:
      - We map ONLY explicit CoW segments (no ACA/PCA synonym mapping).
      - BA is split across A59 and A56.
    """
    return {
        ("R", "ICA"): ["A12"],
        ("L", "ICA"): ["A16"],

        ("R", "MCA"): ["A70"],
        ("L", "MCA"): ["A73"],

        ("R", "A1"): ["A68"],
        ("L", "A1"): ["A69"],
        ("R", "A2"): ["A76"],
        ("L", "A2"): ["A78"],

        ("R", "P1"): ["A60"],
        ("L", "P1"): ["A61"],
        ("R", "P2"): ["A64"],
        ("L", "P2"): ["A65"],

        ("R", "Pcom"): ["A62"],
        ("L", "Pcom"): ["A63"],

        (None, "Acom"): ["A77"],
        (None, "BA"): ["A59", "A56"],
    }


# -------------------------
# Geometry injection helpers
# -------------------------

REQUIRED_ARTERIAL_COLUMNS = {"ID", "name", "length[SI]", "start_diameter[SI]", "end_diameter[SI]"}

def assert_arterial_schema(df: pd.DataFrame):
    missing = [c for c in REQUIRED_ARTERIAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "arterial.csv schema mismatch. Missing columns: " + ", ".join(missing)
            + "\nAvailable columns: " + ", ".join(df.columns)
        )


def apply_geometry(df: pd.DataFrame, fb_id: str, length_m: float, diameter_m: float) -> Optional[Dict[str, Any]]:
    idxs = df.index[df["ID"] == fb_id].tolist()
    if not idxs:
        return None
    i = idxs[0]

    old_length = float(df.loc[i, "length[SI]"])
    old_d1 = float(df.loc[i, "start_diameter[SI]"])
    old_d2 = float(df.loc[i, "end_diameter[SI]"])
    name = str(df.loc[i, "name"])

    df.loc[i, "length[SI]"] = float(length_m)
    df.loc[i, "start_diameter[SI]"] = float(diameter_m)
    df.loc[i, "end_diameter[SI]"] = float(diameter_m)

    # thickness untouched

    return {
        "Action": "inject",
        "FirstBlood_ID": fb_id,
        "Name": name,
        "Old_length_mm": old_length * 1000.0,
        "New_length_mm": length_m * 1000.0,
        "Old_diameter_mm": old_d1 * 1000.0,
        "New_diameter_mm": diameter_m * 1000.0,
        "Old_end_diameter_mm": old_d2 * 1000.0,
        "New_end_diameter_mm": diameter_m * 1000.0,
    }


def split_length_by_template(df: pd.DataFrame, ids: List[str], total_length_m: float) -> List[float]:
    template_lengths = []
    for fb_id in ids:
        idxs = df.index[df["ID"] == fb_id].tolist()
        template_lengths.append(float(df.loc[idxs[0], "length[SI]"]) if idxs else 0.0)
    s = float(np.sum(template_lengths))
    if s <= 1e-12:
        return [total_length_m / len(ids) for _ in ids]
    return [total_length_m * (tl / s) for tl in template_lengths]


def occlude_vessel(df: pd.DataFrame, fb_id: str, min_diameter_m: float) -> Optional[Dict[str, Any]]:
    idxs = df.index[df["ID"] == fb_id].tolist()
    if not idxs:
        return None
    i = idxs[0]
    name = str(df.loc[i, "name"])
    old_d1 = float(df.loc[i, "start_diameter[SI]"])
    old_d2 = float(df.loc[i, "end_diameter[SI]"])

    df.loc[i, "start_diameter[SI]"] = float(min_diameter_m)
    df.loc[i, "end_diameter[SI]"] = float(min_diameter_m)

    return {
        "Action": "occlude_absent",
        "FirstBlood_ID": fb_id,
        "Name": name,
        "Old_diameter_mm": old_d1 * 1000.0,
        "New_diameter_mm": min_diameter_m * 1000.0,
        "Old_end_diameter_mm": old_d2 * 1000.0,
        "New_end_diameter_mm": min_diameter_m * 1000.0,
    }


# -------------------------
# Variant parsing -> absent vessel keys (STRICT + SAFE)
# -------------------------

VARIANT_ALLOWED_NAMES = {"A1", "A2", "Acom", "Pcom", "P1", "P2", "ICA", "MCA", "BA"}
VARIANT_ABSENCE_GROUPS = {"anterior", "posterior"}  # ONLY these are treated as present/absent

def extract_absent_variant_keys(variants_json: Any) -> Tuple[List[Tuple[Optional[str], str]], List[Dict[str, Any]]]:
    """
    Returns:
      absent_keys: list of (side, canon) where canon is in VARIANT_ALLOWED_NAMES
      ignored: list of logs for keys ignored (unknown groups, unknown labels, etc.)

    Rules:
      - ONLY groups in VARIANT_ABSENCE_GROUPS are used for absence (False => absent)
      - fetal + fenestration are ignored for absence decisions
      - keys like "3rd-A2" are ignored (not in Abel_ref2)
    """
    absent: List[Tuple[Optional[str], str]] = []
    ignored: List[Dict[str, Any]] = []

    if not isinstance(variants_json, dict):
        return absent, ignored

    def parse_key(k: Any) -> Tuple[Optional[str], str, bool]:
        """
        Returns (side, canon, ok)
        ok=False if canon not allowed or malformed.
        """
        if not isinstance(k, str):
            return (None, str(k), False)
        ks = k.strip()

        side: Optional[str] = None
        name = ks

        if "-" in ks:
            s, n = ks.split("-", 1)
            s = s.strip()
            n = n.strip()
            if s in ("R", "L"):
                side = s
                name = n
            else:
                # e.g. "3rd-A2" -> side not R/L
                return (None, ks, False)

        canon = normalize_segment_name(name)

        if canon not in VARIANT_ALLOWED_NAMES:
            return (side, canon, False)

        return (side, canon, True)

    for group, blob in variants_json.items():
        if group not in VARIANT_ABSENCE_GROUPS:
            # ignored group (fetal/fenestration/etc.)
            continue
        if not isinstance(blob, dict):
            continue

        for k, present in blob.items():
            # only treat explicit False as absent
            if present is False:
                side, canon, ok = parse_key(k)
                if ok:
                    absent.append((side, canon))
                else:
                    ignored.append({"group": group, "key": str(k), "reason": "ignored_absence_key_not_supported"})

    # de-dup
    out = []
    seen = set()
    for a in absent:
        if a not in seen:
            out.append(a)
            seen.add(a)
    return out, ignored


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", default=None, help="patient id, e.g. 025")
    ap.add_argument("--template_model", default="Abel_ref2", help="template model folder under <repo_root>/models")
    ap.add_argument("--out_model_name", default=None, help="output model name under <repo_root>/models")
    ap.add_argument("--force", action="store_true", help="overwrite output model folder if it exists")
    ap.add_argument("--min_absent_diameter_mm", type=float, default=0.001, help="occlusion diameter for absent vessels [mm]")
    args = ap.parse_args()

    pid = args.pid
    if pid is None or str(pid).strip() == "":
        pid = input("Enter patient id (e.g. 025): ").strip()
    pid = str(pid).zfill(3)

    out_model_name = args.out_model_name or f"patient_{pid}"

    repo_root = get_repo_root()
    data_root = repo_root / "data"
    template_dir = repo_root / "models" / args.template_model
    out_dir = repo_root / "models" / out_model_name

    if not template_dir.exists():
        raise FileNotFoundError(f"Template model not found: {template_dir}")
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    modality, feature_path, nodes_path, variant_path = find_patient_files(data_root, pid)

    print("=" * 78)
    print("PATIENT COW INJECTION (STRICT, no synonyms)")
    print("=" * 78)
    print(f"repo_root:      {repo_root}")
    print(f"pid:            {pid}")
    print(f"modality:       {modality}")
    print(f"feature file:   {feature_path}")
    print(f"nodes file:     {nodes_path}")
    print(f"variant file:   {variant_path if variant_path else '(none)'}")
    print(f"template model: {template_dir}")
    print(f"output model:   {out_dir}")
    print("-" * 78)

    features = load_json(feature_path)
    nodes = load_json(nodes_path)
    variants = load_json(variant_path) if variant_path else None

    id_to_xyz, node_side_hint, r_x, l_x = flatten_nodes(nodes)
    right_is_pos_x = infer_right_is_positive_x(r_x, l_x)
    print(f"Right is positive x? {right_is_pos_x} (from explicit R/L hints if available)")

    segs = parse_patient_features(features)
    print(f"Parsed {len(segs)} raw feature segments (pre-filter).")

    fb_map = build_firstblood_mapping()

    # Keep only segments that are canonical (strict) OR BA/Acom/Pcom etc.
    candidates: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for s in segs:
        canon = normalize_segment_name(s["raw_name"])

        # Strict: only keep segments we can map (plus BA/Acom which are side=None)
        # Everything else becomes "unmapped patient keys"
        if canon not in CANONICAL:
            candidates.append({**s, "canon": canon, "side": None, "reason": "noncanonical_ignored_for_mapping"})
            continue

        side = side_from_endpoints(id_to_xyz, node_side_hint, s["start_id"], s["end_id"], right_is_pos_x)

        length_m = float(s["length_mm"]) / 1000.0
        diameter_m = (2.0 * float(s["radius_mm"])) / 1000.0

        if canon in {"BA", "Acom"}:
            key = (None, canon)
        else:
            if side is None:
                skipped.append({
                    "Reason": "side_unknown_for_side_specific_vessel",
                    "raw_name": s["raw_name"],
                    "canon": canon,
                    "start_id": s["start_id"],
                    "end_id": s["end_id"],
                    "length_mm": s["length_mm"],
                    "radius_mm": s["radius_mm"],
                })
                continue
            key = (side, canon)

        candidates.append({
            "key": key,
            "raw_name": s["raw_name"],
            "canon": canon,
            "side": side,
            "start_id": s["start_id"],
            "end_id": s["end_id"],
            "length_m": length_m,
            "diameter_m": diameter_m,
        })

    # Build patient_geom by selecting ONE best candidate per (side, canon)
    # Selection rule: choose the candidate with the LARGEST length (stable, but per side+canon only)
    patient_geom: Dict[Tuple[Optional[str], str], Dict[str, Any]] = {}
    for c in candidates:
        if "key" not in c:
            continue
        key = c["key"]
        if key not in patient_geom or float(c["length_m"]) > float(patient_geom[key]["length_m"]):
            patient_geom[key] = c

    print(f"Kept {len(patient_geom)} unique (side, canon) geometry entries.")
    print(f"Skipped (side unknown): {len(skipped)}")
    print("-" * 78)

    # Prepare output folder
    if out_dir.exists():
        if not args.force:
            raise FileExistsError(f"Output model exists: {out_dir}\nUse --force to overwrite.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    template_arterial = template_dir / "arterial.csv"
    if not template_arterial.exists():
        raise FileNotFoundError(f"Missing arterial.csv in template: {template_arterial}")

    # Copy ALL template csv except arterial.csv
    for src in template_dir.glob("*.csv"):
        if src.name == "arterial.csv":
            continue
        shutil.copy(src, out_dir / src.name)

    # Load arterial template
    df = pd.read_csv(template_arterial)
    assert_arterial_schema(df)

    modifications: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    unmapped_patient_keys: List[Dict[str, Any]] = []

    # Inject BA (split)
    ba_key = (None, "BA")
    if ba_key in patient_geom:
        fb_ids = fb_map[ba_key]
        total_len = float(patient_geom[ba_key]["length_m"])
        diam = float(patient_geom[ba_key]["diameter_m"])
        split_lens = split_length_by_template(df, fb_ids, total_len)

        for fb_id, L in zip(fb_ids, split_lens):
            mod = apply_geometry(df, fb_id, L, diam)
            if mod:
                mod.update({"Patient_key": "BA", "Split_mode": "proportional_to_template"})
                modifications.append(mod)
            else:
                missing.append({"Patient_key": "BA", "FirstBlood_ID": fb_id, "Reason": "ID not found in arterial.csv"})

    # Inject everything else
    for (side, canon), g in patient_geom.items():
        if canon == "BA":
            continue

        map_key = (None, canon) if canon == "Acom" else (side, canon)
        pk = canon if side is None else f"{side}_{canon}"

        if map_key not in fb_map:
            unmapped_patient_keys.append({"Patient_key": pk, "Reason": "no_mapping_in_fb_map", "raw_name": g.get("raw_name")})
            continue

        for fb_id in fb_map[map_key]:
            mod = apply_geometry(df, fb_id, float(g["length_m"]), float(g["diameter_m"]))
            if mod:
                mod.update({"Patient_key": pk, "Split_mode": ""})
                modifications.append(mod)
            else:
                missing.append({"Patient_key": pk, "FirstBlood_ID": fb_id, "Reason": "ID not found in arterial.csv"})

    # Absent vessel policy (STRICT)
    absent_keys, ignored_variant_keys = extract_absent_variant_keys(variants)
    min_absent_diam_m = float(args.min_absent_diameter_mm) / 1000.0

    variant_conflicts: List[Dict[str, Any]] = []

    # Build set of measured keys we actually have (so "measured beats variants")
    measured_keys = set(patient_geom.keys())

    for akey in absent_keys:
        pk = akey[1] if akey[0] is None else f"{akey[0]}_{akey[1]}"

        # If measured exists -> conflict, do NOT occlude
        if akey in measured_keys:
            variant_conflicts.append({
                "Patient_key": pk,
                "Reason": "variant_marks_absent_but_measured_exists",
            })
            continue

        if akey not in fb_map:
            # should be rare due to whitelist, but keep safe
            ignored_variant_keys.append({"group": "anterior/posterior", "key": pk, "reason": "absent_key_not_in_fb_map"})
            continue

        for fb_id in fb_map[akey]:
            oc = occlude_vessel(df, fb_id, min_diameter_m=min_absent_diam_m)
            if oc:
                oc.update({"Patient_key": pk, "Split_mode": ""})
                modifications.append(oc)
            else:
                missing.append({"Patient_key": pk, "FirstBlood_ID": fb_id, "Reason": "ID not found in arterial.csv (absent occlusion)"})

    # Save outputs
    df.to_csv(out_dir / "arterial.csv", index=False)

    pd.DataFrame(modifications).to_csv(out_dir / "modifications_log.csv", index=False)
    pd.DataFrame(missing).to_csv(out_dir / "missing_mapping_log.csv", index=False)
    pd.DataFrame(skipped).to_csv(out_dir / "skipped_segments_log.csv", index=False)
    pd.DataFrame(unmapped_patient_keys).to_csv(out_dir / "unmapped_patient_keys_log.csv", index=False)
    pd.DataFrame(ignored_variant_keys).to_csv(out_dir / "variant_ignored_keys_log.csv", index=False)
    pd.DataFrame(variant_conflicts).to_csv(out_dir / "variant_conflicts_log.csv", index=False)

    print("Injection complete.")
    print(f"Modified entries (inject + occlude): {len(modifications)}")
    print(f"Missing mappings:                  {len(missing)}")
    print(f"Skipped segments (side unknown):   {len(skipped)}")
    print(f"Unmapped patient keys:             {len(unmapped_patient_keys)}")
    print(f"Ignored variant keys:              {len(ignored_variant_keys)}")
    print(f"Variant conflicts (measured wins): {len(variant_conflicts)}")

    if absent_keys:
        print("-" * 78)
        print("ABSENT VESSEL POLICY APPLIED (occlusion):")
        print(f"  absent keys flagged (strict): {len(absent_keys)}")
        print(f"  occlusion diameter:           {args.min_absent_diameter_mm} mm")
        for k in absent_keys:
            print(f"  - {k}")

    print("-" * 78)
    print("Model folder created:")
    print(f"  {out_dir}")
    print("Next (example):")
    print(f"  cd {repo_root}/projects/simple_run")
    print(f"  ./simple_run.out {out_model_name}")


if __name__ == "__main__":
    main()