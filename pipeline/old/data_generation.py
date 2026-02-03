#!/usr/bin/env python3
"""
Patient-specific Circle of Willis geometry injection for first_blood (Abel_ref2 template).

Fixes implemented (per your request)
1) Delete synonym mapping (NO ACA->A1, NO PCA->P1)
2) Normalize segment names + skip "whole vessel" labels when parts exist in raw data:
     - If A1/A2 exist -> ignore ACA
     - If P1/P2 exist -> ignore PCA
     - If C6/C7 exist -> ignore ICA (and by default we do not inject ICA at all)
   This is necessary because your raw features include both whole + parts. :contentReference[oaicite:3]{index=3}
3) Variant occlusion:
     - only use variants groups: anterior, posterior (absence flags)
     - ignore fetal, fenestration groups (not "absence") :contentReference[oaicite:4]{index=4}
     - measured geometry overrides variants (never occlude measured)
4) No invented wall thickness: thickness columns are NOT overwritten.

Run from: <repo_root>/pipeline
Writes:     <repo_root>/models/patient_<pid>/
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Repo-relative paths
# -------------------------

def get_repo_root() -> Path:
    # script expected in <repo_root>/pipeline/
    return Path(__file__).resolve().parent.parent


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
    Supports:
      topcow_<modality>_<pid>.json
      feature_<modality>_<pid>.json / nodes_<modality>_<pid>.json / variant_<modality>_<pid>.json
    Preference: ct then mr
    """
    pid = str(pid).zfill(3)
    for modality in ["ct", "mr"]:
        feat_a = data_root / "cow_features" / f"topcow_{modality}_{pid}.json"
        nods_a = data_root / "cow_nodes" / f"topcow_{modality}_{pid}.json"
        var_a  = data_root / "cow_variants" / f"topcow_{modality}_{pid}.json"

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
# Nodes + side inference
# -------------------------

def flatten_nodes(nodes_json: Any) -> Tuple[Dict[int, np.ndarray], List[float], List[float]]:
    """
    Collect dicts like {"id": ..., "coords": [x,y,z]}
    Track explicit R-/L- hints if present in ancestor keys.
    """
    id_to_xyz: Dict[int, np.ndarray] = {}
    r_x: List[float] = []
    l_x: List[float] = []

    def on_dict(d, path):
        if "id" in d and "coords" in d:
            try:
                nid = int(d["id"])
                coords = d["coords"]
                if isinstance(coords, (list, tuple)) and len(coords) >= 3:
                    xyz = np.array([float(coords[0]), float(coords[1]), float(coords[2])], dtype=float)
                    id_to_xyz[nid] = xyz

                    # Only trust explicit "R-" / "L-" keys as strong hints
                    side_hint = None
                    for p in reversed(path):
                        if isinstance(p, str):
                            if p.startswith("R-"):
                                side_hint = "R"
                                break
                            if p.startswith("L-"):
                                side_hint = "L"
                                break

                    if side_hint == "R":
                        r_x.append(float(xyz[0]))
                    elif side_hint == "L":
                        l_x.append(float(xyz[0]))
            except Exception:
                return

    _walk_json(nodes_json, on_dict=on_dict)
    return id_to_xyz, r_x, l_x


def infer_right_is_positive_x(r_x: List[float], l_x: List[float]) -> bool:
    if len(r_x) == 0 or len(l_x) == 0:
        return True
    return float(np.mean(r_x)) > float(np.mean(l_x))


def side_from_endpoints(
    id_to_xyz: Dict[int, np.ndarray],
    start_id: int,
    end_id: int,
    right_is_positive_x: bool
) -> Optional[str]:
    xs: List[float] = []
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
    Find entries containing:
      - segment: {start, end}
      - length (mm)
      - radius: {mean} (mm)

    Uses last non-numeric key from JSON path as raw_name.
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
                "raw_name": raw_name,
                "start_id": start_id,
                "end_id": end_id,
                "length_mm": length_mm,
                "radius_mm": radius_mm,
            }
        )

    _walk_json(features_json, on_dict=on_dict)
    return segs


def normalize_segment_name(raw_name: str) -> str:
    """
    Normalize to a conservative canonical vocabulary.
    IMPORTANT: we keep ACA/PCA/ICA as-is (not mapped to A1/P1 etc) to avoid synonyms.
    """
    name = str(raw_name).strip()

    known = {
        "A1", "A2", "Acom", "Pcom", "MCA", "BA", "P1", "P2",
        "ACA", "PCA", "ICA", "C6", "C7"
    }
    if name in known:
        return name

    low = name.lower()
    if low in {"basilar", "basilar artery"}:
        return "BA"
    if low in {"acom", "a-com", "anterior communicating", "anterior_comm"}:
        return "Acom"
    if low in {"pcom", "p-com", "posterior communicating", "posterior_comm"}:
        return "Pcom"
    if low in {"middle cerebral", "middle_cerebral"}:
        return "MCA"
    if low in {"internal carotid", "internal_carotid"}:
        return "ICA"

    return name


def compute_present_parts(segs: List[Dict[str, Any]]) -> Dict[str, bool]:
    """
    Detect whether the raw file contains 'parts' that make a 'whole' redundant.
    This is purely data-driven (no biology needed).
    """
    present = set(normalize_segment_name(s["raw_name"]) for s in segs)

    return {
        "has_A_parts": ("A1" in present) or ("A2" in present),
        "has_P_parts": ("P1" in present) or ("P2" in present),
        "has_ICA_parts": ("C6" in present) or ("C7" in present),
    }


def should_skip_canon(canon: str, parts: Dict[str, bool]) -> bool:
    """
    Skip whole-vessel labels when parts exist in the raw data.
    Your patient file includes both whole and parts (e.g., ACA + A1/A2). :contentReference[oaicite:5]{index=5}
    """
    if canon == "ACA" and parts["has_A_parts"]:
        return True
    if canon == "PCA" and parts["has_P_parts"]:
        return True
    if canon == "ICA" and parts["has_ICA_parts"]:
        return True
    return False


# -------------------------
# FirstBlood mapping (NO synonyms)
# -------------------------

def build_firstblood_mapping() -> Dict[Tuple[Optional[str], str], List[str]]:
    """
    Mapping into Abel_ref2 IDs.
    IMPORTANT: no synonym entries (no ACA/PCA mappings).
    """
    return {
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

        # NOTE: ICA intentionally not injected by default
    }


# -------------------------
# Geometry injection helpers
# -------------------------

REQUIRED_ARTERIAL_COLUMNS = {
    "ID",
    "name",
    "length[SI]",
    "start_diameter[SI]",
    "end_diameter[SI]",
}


def assert_arterial_schema(df: pd.DataFrame):
    missing = [c for c in REQUIRED_ARTERIAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "arterial.csv schema mismatch. Missing columns: "
            + ", ".join(missing)
            + "\nAvailable columns: " + ", ".join(df.columns)
        )


def apply_geometry(df_arterial: pd.DataFrame, fb_id: str, length_m: float, diameter_m: float) -> Optional[Dict[str, Any]]:
    idxs = df_arterial.index[df_arterial["ID"] == fb_id].tolist()
    if not idxs:
        return None
    i = idxs[0]

    old_length = float(df_arterial.loc[i, "length[SI]"])
    old_d1 = float(df_arterial.loc[i, "start_diameter[SI]"])
    old_d2 = float(df_arterial.loc[i, "end_diameter[SI]"])
    name = str(df_arterial.loc[i, "name"])

    df_arterial.loc[i, "length[SI]"] = float(length_m)
    df_arterial.loc[i, "start_diameter[SI]"] = float(diameter_m)
    df_arterial.loc[i, "end_diameter[SI]"] = float(diameter_m)

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


def split_length_by_template(df_arterial: pd.DataFrame, ids: List[str], total_length_m: float) -> List[float]:
    template_lengths = []
    for fb_id in ids:
        idxs = df_arterial.index[df_arterial["ID"] == fb_id].tolist()
        template_lengths.append(float(df_arterial.loc[idxs[0], "length[SI]"]) if idxs else 0.0)
    s = float(np.sum(template_lengths))
    if s <= 1e-12:
        return [total_length_m / len(ids) for _ in ids]
    fracs = [tl / s for tl in template_lengths]
    return [total_length_m * f for f in fracs]


def occlude_vessel(df_arterial: pd.DataFrame, fb_id: str, min_diameter_m: float) -> Optional[Dict[str, Any]]:
    idxs = df_arterial.index[df_arterial["ID"] == fb_id].tolist()
    if not idxs:
        return None
    i = idxs[0]
    name = str(df_arterial.loc[i, "name"])
    old_d1 = float(df_arterial.loc[i, "start_diameter[SI]"])
    old_d2 = float(df_arterial.loc[i, "end_diameter[SI]"])

    df_arterial.loc[i, "start_diameter[SI]"] = float(min_diameter_m)
    df_arterial.loc[i, "end_diameter[SI]"] = float(min_diameter_m)

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
# Variants -> absent vessel keys
# -------------------------

def extract_absent_variant_keys(variants_json: Any) -> List[Tuple[Optional[str], str]]:
    """
    Only interpret variants groups anterior/posterior as "absence".
    Ignore fetal/fenestration (not absence in your file). :contentReference[oaicite:6]{index=6}
    Keys returned in (side, canon) form compatible with fb_map.
    """
    absent: List[Tuple[Optional[str], str]] = []
    if not isinstance(variants_json, dict):
        return absent

    allowed_groups = {"anterior", "posterior"}  # only these are absence flags for your dataset

    def parse_key(k: Any) -> Tuple[Optional[str], str]:
        if not isinstance(k, str):
            return (None, str(k))
        ks = k.strip()
        if "-" in ks:
            s, n = ks.split("-", 1)
            s = s.strip()
            n = n.strip()
            if s in ("R", "L"):
                return (s, normalize_segment_name(n))
        return (None, normalize_segment_name(ks))

    for group, blob in variants_json.items():
        if group not in allowed_groups:
            continue
        if not isinstance(blob, dict):
            continue
        for k, present in blob.items():
            if present is False:
                absent.append(parse_key(k))

    # de-dup
    seen = set()
    uniq: List[Tuple[Optional[str], str]] = []
    for a in absent:
        if a not in seen:
            uniq.append(a)
            seen.add(a)
    return uniq


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
    print("PATIENT COW INJECTION (repo-relative)")
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

    id_to_xyz, r_x, l_x = flatten_nodes(nodes)
    right_is_pos_x = infer_right_is_positive_x(r_x, l_x)
    print(f"Right is positive x? {right_is_pos_x}")

    segs = parse_patient_features(features)
    print(f"Parsed {len(segs)} raw feature segments (pre-filter).")

    parts = compute_present_parts(segs)
    if parts["has_A_parts"]:
        print("Detected A1/A2 in raw data -> will ignore ACA whole-vessel entries.")
    if parts["has_P_parts"]:
        print("Detected P1/P2 in raw data -> will ignore PCA whole-vessel entries.")
    if parts["has_ICA_parts"]:
        print("Detected C6/C7 in raw data -> will ignore ICA whole-vessel entries.")

    # patient geometry keyed by (side, canon)
    patient_geom: Dict[Tuple[Optional[str], str], Dict[str, Any]] = {}
    skipped: List[Dict[str, Any]] = []

    for s in segs:
        canon = normalize_segment_name(s["raw_name"])

        if should_skip_canon(canon, parts):
            continue

        # We only inject a safe subset into Abel_ref2 CoW mapping
        # Skip labels we don't map (like ACA/PCA/ICA/C6/C7 unless you later add mapping)
        # This prevents accidental wrong-span injections.
        if canon in {"ACA", "PCA", "ICA", "C6", "C7"}:
            continue

        side = side_from_endpoints(id_to_xyz, s["start_id"], s["end_id"], right_is_pos_x)

        length_m = float(s["length_mm"]) / 1000.0
        diameter_m = (2.0 * float(s["radius_mm"])) / 1000.0

        if canon in {"BA", "Acom"}:
            key = (None, canon)
        else:
            key = (side, canon)

        # Side required but unknown -> log and skip
        if canon not in {"BA", "Acom"} and side is None:
            skipped.append({
                "Reason": "side_unknown",
                "raw_name": s["raw_name"],
                "canon": canon,
                "start_id": s["start_id"],
                "end_id": s["end_id"],
                "length_mm": s["length_mm"],
                "radius_mm": s["radius_mm"],
            })
            continue

        # Keep the longest (simple, stable). If you later want, we can replace with sum/weighted mean.
        if key not in patient_geom or length_m > patient_geom[key]["length_m"]:
            patient_geom[key] = {
                "length_m": length_m,
                "diameter_m": diameter_m,
                "raw_name": s["raw_name"],
                "start_id": s["start_id"],
                "end_id": s["end_id"],
                "side": side,
                "canon": canon,
            }

    print(f"Kept {len(patient_geom)} injectable entries after filtering.")
    if skipped:
        print(f"Skipped {len(skipped)} segments due to unknown side (logged).")
    print("-" * 78)

    # Prepare output folder
    if out_dir.exists():
        if not args.force:
            raise FileExistsError(f"Output model exists: {out_dir}\nUse --force to overwrite.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy ALL csv from template except arterial.csv
    template_arterial = template_dir / "arterial.csv"
    if not template_arterial.exists():
        raise FileNotFoundError(f"Missing arterial.csv in template: {template_arterial}")

    for src in template_dir.glob("*.csv"):
        if src.name == "arterial.csv":
            continue
        shutil.copy(src, out_dir / src.name)

    # Load arterial template and inject
    df = pd.read_csv(template_arterial)
    assert_arterial_schema(df)

    fb_map = build_firstblood_mapping()

    modifications: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    unmapped_patient_keys: List[Dict[str, Any]] = []
    variant_conflicts: List[Dict[str, Any]] = []

    # Track which patient keys are "measured present"
    measured_present_keys = set(patient_geom.keys())

    # BA split
    ba_key = (None, "BA")
    if ba_key in patient_geom and ba_key in fb_map:
        fb_ids = fb_map[ba_key]
        total_len = patient_geom[ba_key]["length_m"]
        diam = patient_geom[ba_key]["diameter_m"]
        split_lens = split_length_by_template(df, fb_ids, total_len)
        for fb_id, L in zip(fb_ids, split_lens):
            mod = apply_geometry(df, fb_id, L, diam)
            if mod:
                mod.update({"Patient_key": "BA", "Split_mode": "proportional_to_template"})
                modifications.append(mod)
            else:
                missing.append({"Patient_key": "BA", "FirstBlood_ID": fb_id, "Reason": "ID not found in arterial.csv"})

    # Inject other segments
    for (side, canon), g in patient_geom.items():
        if canon == "BA":
            continue
        map_key = (None, canon) if canon == "Acom" else (side, canon)

        pk = canon if side is None else f"{side}_{canon}"
        if map_key not in fb_map:
            unmapped_patient_keys.append({"Patient_key": pk, "Reason": "no_mapping_in_fb_map", "raw_name": g.get("raw_name")})
            continue

        for fb_id in fb_map[map_key]:
            mod = apply_geometry(df, fb_id, g["length_m"], g["diameter_m"])
            if mod:
                mod.update({"Patient_key": pk, "Split_mode": ""})
                modifications.append(mod)
            else:
                missing.append({"Patient_key": pk, "FirstBlood_ID": fb_id, "Reason": "ID not found in arterial.csv"})

    # Absent vessel policy (variants -> occlusion), but measured overrides variants
    absent_keys = extract_absent_variant_keys(variants)
    min_absent_diam_m = float(args.min_absent_diameter_mm) / 1000.0

    if absent_keys:
        for akey in absent_keys:
            # if we measured it, do NOT occlude; log conflict
            if akey in measured_present_keys:
                variant_conflicts.append({
                    "Patient_key": f"{akey[0]}_{akey[1]}" if akey[0] else akey[1],
                    "Reason": "variant_absent_but_measured_present",
                })
                continue

            if akey not in fb_map:
                modifications.append({
                    "Action": "absent_flag_unmapped",
                    "FirstBlood_ID": "",
                    "Name": "",
                    "Old_length_mm": "",
                    "New_length_mm": "",
                    "Old_diameter_mm": "",
                    "New_diameter_mm": "",
                    "Old_end_diameter_mm": "",
                    "New_end_diameter_mm": "",
                    "Patient_key": f"{akey[0]}_{akey[1]}" if akey[0] else akey[1],
                    "Split_mode": "",
                })
                continue

            for fb_id in fb_map[akey]:
                oc = occlude_vessel(df, fb_id, min_diameter_m=min_absent_diam_m)
                if oc:
                    oc.update({"Patient_key": f"{akey[0]}_{akey[1]}" if akey[0] else akey[1], "Split_mode": ""})
                    modifications.append(oc)
                else:
                    missing.append({
                        "Patient_key": f"{akey[0]}_{akey[1]}" if akey[0] else akey[1],
                        "FirstBlood_ID": fb_id,
                        "Reason": "ID not found in arterial.csv (absent occlusion)",
                    })

    # Save outputs
    df.to_csv(out_dir / "arterial.csv", index=False)

    pd.DataFrame(modifications).to_csv(out_dir / "modifications_log.csv", index=False)
    pd.DataFrame(missing).to_csv(out_dir / "missing_mapping_log.csv", index=False)
    pd.DataFrame(skipped).to_csv(out_dir / "skipped_segments_log.csv", index=False)
    pd.DataFrame(unmapped_patient_keys).to_csv(out_dir / "unmapped_patient_keys_log.csv", index=False)
    pd.DataFrame(variant_conflicts).to_csv(out_dir / "variant_conflicts_log.csv", index=False)

    print("Injection complete.")
    print(f"Modified entries (inject + occlude): {len(modifications)}")
    print(f"Missing mappings:                  {len(missing)}")
    print(f"Skipped segments (side unknown):   {len(skipped)}")
    print(f"Unmapped patient keys:             {len(unmapped_patient_keys)}")
    print(f"Variant conflicts (measured wins): {len(variant_conflicts)}")

    if absent_keys:
        print("-" * 78)
        print("ABSENT VESSEL POLICY APPLIED (occlusion):")
        print("  groups used: anterior, posterior")
        print(f"  absent keys flagged: {len(absent_keys)}")
        print(f"  occlusion diameter:  {args.min_absent_diameter_mm} mm")
        for k in absent_keys:
            print(f"  - {k}")

    if variant_conflicts:
        print("-" * 78)
        print("VARIANT CONFLICTS (not occluded because measured present):")
        for c in variant_conflicts:
            print(f"  - {c['Patient_key']}")

    print("-" * 78)
    print("Model folder created:")
    print(f"  {out_dir}")
    print("Next (example):")
    print(f"  cd {repo_root}/projects/simple_run")
    print(f"  ./simple_run.out {out_model_name}")


if __name__ == "__main__":
    main()
