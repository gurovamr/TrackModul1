#!/usr/bin/env python3
"""
Modular Circle of Willis geometry injection for FirstBlood.

Run from:
  ~/first_blood/pipeline

Reads patient raw data from:
  ~/first_blood/data/
    cow_features/topcow_<modality>_<pid>.json
    cow_nodes/topcow_<modality>_<pid>.json
    cow_variants/topcow_<modality>_<pid>.json   (optional)

Creates a new model in:
  ~/first_blood/models/patient_<pid>/

Copies from template (default Abel_ref2) ONLY:
  - main.csv
  - heart_kim_lit.csv
  - p1.csv ... p47.csv

Generates:
  - arterial.csv (template arterial structure + injected patient geometry)
  - modifications_log.csv
  - missing_mapping_log.csv

Notes:
- Side (R/L) is inferred from node x-coordinates when possible, using any
  explicit group labels like "R-*" / "L-*". If unavailable, falls back to x>=0.
- Basilar (BA) patient segment is split across two FirstBlood segments (A59, A56)
  by proportional-to-template-length split (no invented total length).
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# -------------------------
# Utilities
# -------------------------

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def find_patient_files(data_root: Path, pid: str):
    """
    Your files are named:
      cow_features/topcow_<modality>_<pid>.json
      cow_nodes/topcow_<modality>_<pid>.json
      cow_variants/topcow_<modality>_<pid>.json (optional)

    Example (pid=025):
      topcow_ct_025.json

    Preference:
      ct first, then mr (because your current dataset is ct).
    """
    pid = str(pid).zfill(3)

    for modality in ["ct", "mr"]:
        feat = data_root / "cow_features" / f"topcow_{modality}_{pid}.json"
        nods = data_root / "cow_nodes" / f"topcow_{modality}_{pid}.json"
        var = data_root / "cow_variants" / f"topcow_{modality}_{pid}.json"

        if feat.exists() and nods.exists():
            return modality, feat, nods, (var if var.exists() else None)

    raise FileNotFoundError(
        f"Could not find patient files for pid={pid} under {data_root}.\n"
        f"Expected:\n"
        f"  {data_root}/cow_features/topcow_[ct|mr]_{pid}.json\n"
        f"  {data_root}/cow_nodes/topcow_[ct|mr]_{pid}.json\n"
        f"  {data_root}/cow_variants/topcow_[ct|mr]_{pid}.json (optional)\n"
    )


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


def flatten_nodes(nodes_json):
    """
    Robust node flattener:
    - Collects any dicts that look like {"id": ..., "coords": [...]}
    - Tries to detect explicit side labels from ancestor keys containing "R-" or "L-"

    Returns:
      id_to_xyz: dict[int] -> np.array([x,y,z])
      r_x: list of x samples from explicitly R-labeled contexts
      l_x: list of x samples from explicitly L-labeled contexts
    """
    id_to_xyz = {}
    r_x = []
    l_x = []

    def on_dict(d, path):
        if "id" in d and "coords" in d:
            try:
                nid = int(d["id"])
                coords = d["coords"]
                if isinstance(coords, (list, tuple)) and len(coords) >= 3:
                    xyz = np.array([float(coords[0]), float(coords[1]), float(coords[2])], dtype=float)
                    id_to_xyz[nid] = xyz

                    # Look for any ancestor key that starts with R- or L-
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


def infer_right_is_positive_x(r_x, l_x):
    """
    Decide whether "Right" corresponds to positive x or negative x.
    If explicit R/L samples exist, infer from their mean.
    Otherwise assume Right = +x.
    """
    if len(r_x) == 0 or len(l_x) == 0:
        return True

    r_mean = float(np.mean(r_x))
    l_mean = float(np.mean(l_x))
    return r_mean > l_mean


def side_from_endpoints(id_to_xyz, start_id: int, end_id: int, right_is_positive_x: bool):
    """
    Infer side by mean x of endpoints.
    If endpoints missing, returns None.
    """
    xs = []
    if start_id in id_to_xyz:
        xs.append(float(id_to_xyz[start_id][0]))
    if end_id in id_to_xyz:
        xs.append(float(id_to_xyz[end_id][0]))

    if len(xs) == 0:
        return None

    x_mean = float(np.mean(xs))
    if right_is_positive_x:
        return "R" if x_mean >= 0.0 else "L"
    else:
        return "R" if x_mean <= 0.0 else "L"


# -------------------------
# Feature parsing
# -------------------------

def parse_patient_features(features_json):
    """
    Robust feature parser:
    Searches for dict entries that contain:
      - "segment": {"start": <int>, "end": <int>}
      - "length": <float>    (mm)
      - "radius": {"mean": <float>}   (mm)
    And uses the most recent "vessel name" key in the JSON path as raw_name.

    Returns list of dicts:
      [
        {"raw_name": "A1", "start_id": 809, "end_id": 848, "length_mm": 13.3, "radius_mm": 1.6},
        ...
      ]
    """
    segs = []

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

        # Find a vessel-like name from the path (last non-numeric key)
        raw_name = None
        for p in reversed(path):
            if not p.isdigit():
                raw_name = p
                break
        if raw_name is None:
            raw_name = "UNKNOWN"

        # Skip bifurcation-ish names
        if isinstance(raw_name, str) and "bifurcation" in raw_name.lower():
            return

        segs.append(
            {
                "raw_name": str(raw_name),
                "start_id": start_id,
                "end_id": end_id,
                "length_mm": length_mm,
                "radius_mm": radius_mm,
            }
        )

    _walk_json(features_json, on_dict=on_dict)
    return segs


def normalize_segment_name(raw_name: str):
    """
    Map feature names to a canonical set we can map into FirstBlood.
    Expected labels (depending on extraction): A1, A2, Acom, Pcom, MCA, ACA, PCA, BA, P1, P2, ICA
    """
    name = str(raw_name).strip()

    known = {"A1", "A2", "Acom", "Pcom", "MCA", "ACA", "PCA", "BA", "P1", "P2", "ICA"}
    if name in known:
        return name

    low = name.lower()
    if low in {"basilar", "basilar artery"}:
        return "BA"
    if low in {"acom", "a-com", "anterior communicating"}:
        return "Acom"
    if low in {"pcom", "p-com", "posterior communicating"}:
        return "Pcom"

    return name


# -------------------------
# FirstBlood mapping + injection
# -------------------------

def build_firstblood_mapping():
    """
    Hard mapping into FirstBlood IDs (based on your Abel_ref2 CoW IDs).

    BA is split into two template segments:
      A59 and A56
    """
    return {
        ("R", "ICA"): ["A12"],
        ("L", "ICA"): ["A16"],

        ("R", "MCA"): ["A70"],
        ("L", "MCA"): ["A73"],

        ("R", "A1"): ["A68"],
        ("L", "A1"): ["A69"],
        ("R", "ACA"): ["A68"],
        ("L", "ACA"): ["A69"],
        ("R", "A2"): ["A76"],
        ("L", "A2"): ["A78"],

        ("R", "P1"): ["A60"],
        ("L", "P1"): ["A61"],
        ("R", "PCA"): ["A60"],
        ("L", "PCA"): ["A61"],
        ("R", "P2"): ["A64"],
        ("L", "P2"): ["A65"],

        ("R", "Pcom"): ["A62"],
        ("L", "Pcom"): ["A63"],

        (None, "Acom"): ["A77"],

        (None, "BA"): ["A59", "A56"],
    }


def apply_geometry(df_arterial: pd.DataFrame, fb_id: str, length_m: float, diameter_m: float):
    idxs = df_arterial.index[df_arterial["ID"] == fb_id].tolist()
    if len(idxs) == 0:
        return None

    i = idxs[0]
    old_length = float(df_arterial.loc[i, "length[SI]"])
    old_diam = float(df_arterial.loc[i, "start_diameter[SI]"])
    name = str(df_arterial.loc[i, "name"])

    df_arterial.loc[i, "length[SI]"] = float(length_m)
    df_arterial.loc[i, "start_diameter[SI]"] = float(diameter_m)
    df_arterial.loc[i, "end_diameter[SI]"] = float(diameter_m)

    df_arterial.loc[i, "start_thickness[SI]"] = float(diameter_m) * 0.1
    df_arterial.loc[i, "end_thickness[SI]"] = float(diameter_m) * 0.1

    return {
        "FirstBlood_ID": fb_id,
        "Name": name,
        "Old_length_mm": old_length * 1000.0,
        "New_length_mm": length_m * 1000.0,
        "Old_diameter_mm": old_diam * 1000.0,
        "New_diameter_mm": diameter_m * 1000.0,
    }


def split_length_by_template(df_arterial: pd.DataFrame, ids, total_length_m: float):
    """
    Split total_length_m across 'ids' proportionally to the existing template lengths.
    """
    template_lengths = []
    for fb_id in ids:
        idxs = df_arterial.index[df_arterial["ID"] == fb_id].tolist()
        template_lengths.append(float(df_arterial.loc[idxs[0], "length[SI]"]) if idxs else 0.0)

    s = float(np.sum(template_lengths))
    if s <= 1e-12:
        return [total_length_m / len(ids) for _ in ids]

    fracs = [tl / s for tl in template_lengths]
    return [total_length_m * f for f in fracs]


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", default=None, help="patient id, e.g. 025")
    ap.add_argument("--template_model", default="Abel_ref2", help="template model folder under ~/first_blood/models")
    ap.add_argument("--out_model_name", default=None, help="output model name under ~/first_blood/models")
    args = ap.parse_args()

    pid = args.pid
    if pid is None or str(pid).strip() == "":
        pid = input("Enter patient id (e.g. 025): ").strip()
    pid = str(pid).zfill(3)

    template_model = args.template_model
    if args.out_model_name is None:
        out_model_name = f"patient_{pid}"
    else:
        out_model_name = args.out_model_name

    # Your actual data folder is: ~/first_blood/data
    data_root = Path.home() / "first_blood" / "data"

    template_dir = Path.home() / "first_blood" / "models" / template_model
    out_dir = Path.home() / "first_blood" / "models" / out_model_name

    if not template_dir.exists():
        raise FileNotFoundError(f"Template model not found: {template_dir}")
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    modality, feature_path, nodes_path, variant_path = find_patient_files(data_root, pid)

    print("=" * 78)
    print("MODULAR PATIENT COW INJECTION")
    print("=" * 78)
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
    print(f"Parsed {len(segs)} raw feature segments (pre-dedup).")

    # Build patient segment records keyed by (side, canonical_name)
    patient_geom = {}
    for s in segs:
        canon = normalize_segment_name(s["raw_name"])

        # Side from nodes; if nodes missing for this segment, keep None
        side = side_from_endpoints(id_to_xyz, s["start_id"], s["end_id"], right_is_pos_x)

        length_m = float(s["length_mm"]) / 1000.0
        diameter_m = (2.0 * float(s["radius_mm"])) / 1000.0

        # For BA and Acom we do not want side in the key
        if canon in {"BA", "Acom"}:
            key = (None, canon)
        else:
            key = (side, canon)

        # Keep the longest measurement if duplicates appear
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

    print(f"Kept {len(patient_geom)} unique (side, name) entries after dedup.")
    print("-" * 78)

    # Create output model directory fresh
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy only required template files
    copy_names = ["main.csv", "heart_kim_lit.csv"] + [f"p{i}.csv" for i in range(1, 48)]
    for name in copy_names:
        src = template_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing template file: {src}")
        shutil.copy(src, out_dir / name)

    # Load arterial template (structure) from template_dir, but write into out_dir
    template_arterial = template_dir / "arterial.csv"
    if not template_arterial.exists():
        raise FileNotFoundError(f"Missing arterial.csv in template: {template_arterial}")

    df = pd.read_csv(template_arterial)
    fb_map = build_firstblood_mapping()

    modifications = []
    missing = []

    # BA split (None, "BA") across A59 and A56
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

    # Inject all other mapped segments
    for (side, canon), g in patient_geom.items():
        if canon == "BA":
            continue

        map_key = (None, canon) if canon == "Acom" else (side, canon)
        if map_key not in fb_map:
            continue

        for fb_id in fb_map[map_key]:
            mod = apply_geometry(df, fb_id, g["length_m"], g["diameter_m"])
            if mod:
                pk = canon if side is None else f"{side}_{canon}"
                mod.update({"Patient_key": pk, "Split_mode": ""})
                modifications.append(mod)
            else:
                pk = canon if side is None else f"{side}_{canon}"
                missing.append({"Patient_key": pk, "FirstBlood_ID": fb_id, "Reason": "ID not found in arterial.csv"})

    # Variant warnings (optional)
    variant_warnings = []
    if isinstance(variants, dict):
        def vkey_to_tuple(k):
            if isinstance(k, str) and "-" in k:
                s, n = k.split("-", 1)
                if s in ("L", "R"):
                    return (s, n)
            return (None, k)

        for group in ["anterior", "posterior", "fetal", "fenestration"]:
            if group not in variants or not isinstance(variants[group], dict):
                continue
            for k, present in variants[group].items():
                _ = vkey_to_tuple(k)
                if present is False:
                    variant_warnings.append(f"Variant marks absent: {k} ({group})")

    # Save arterial + logs
    arterial_path = out_dir / "arterial.csv"
    df.to_csv(arterial_path, index=False)

    pd.DataFrame(modifications).to_csv(out_dir / "modifications_log.csv", index=False)
    pd.DataFrame(missing).to_csv(out_dir / "missing_mapping_log.csv", index=False)

    print("Injection complete.")
    print(f"Modified segments: {len(modifications)}")
    print(f"Missing mappings:  {len(missing)}")
    if variant_warnings:
        print("-" * 78)
        print("VARIANT FLAGS (review):")
        for w in variant_warnings:
            print(f"  - {w}")

    print("-" * 78)
    print("Model folder created with only:")
    print("  arterial.csv, main.csv, heart_kim_lit.csv, p1..p47.csv (+ logs)")
    print("-" * 78)
    print("Next:")
    print("  cd ~/first_blood/projects/simple_run")
    print(f"  ./simple_run.out {out_model_name}")


if __name__ == "__main__":
    main()
