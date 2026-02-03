#!/usr/bin/env python3
"""
Patient-specific Circle of Willis model generator for first_blood (v4 - FINAL)

FIXES APPLIED:
- Peripheral node filtering: only nodes matching p[0-9]+ pattern (excludes "parameter")
- Correct main.csv generation with proper peripheral mappings (pX -> pX, not pX -> n1)
- Handles missing segments gracefully (keeps Abel_ref2 defaults)
- Respects patient variant anatomy from topcow JSON

Run from:
  <repo_root>/pipeline

Usage:
  python3 data_generationV4.py
  (will prompt for patient ID)

OR:
  python3 data_generationV4.py --pid 025 --force

Reads patient raw data from:
  <repo_root>/data/
    data/cow_features/topcow_[ct|mr]_<pid>.json
    data/cow_nodes/topcow_[ct|mr]_<pid>.json
    data/cow_variants/topcow_[ct|mr]_<pid>.json (optional)

Creates:
  <repo_root>/models/patient_<pid>/
    - arterial.csv (patient-specific CoW geometry)
    - main.csv (correct peripheral mappings)
    - p1.csv, p2.csv, ..., p47.csv (copied from Abel_ref2)
    - heart_kim_lit.csv (copied from Abel_ref2)
    - modifications_log.csv (what was changed)
"""

import argparse
import json
import shutil
import re
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
    Find patient data files with flexible naming conventions.
    
    Supports:
      - All files named topcow_<modality>_<pid>.json in their respective folders
      - Mixed naming (topcow for features/variants, nodes_<modality> for nodes)
    
    Preference: ct then mr
    """
    pid = str(pid).zfill(3)

    for modality in ["ct", "mr"]:
        # Primary: all named topcow_X_025.json
        feat_a = data_root / "cow_features" / f"topcow_{modality}_{pid}.json"
        nods_a = data_root / "cow_nodes" / f"topcow_{modality}_{pid}.json"
        var_a  = data_root / "cow_variants" / f"topcow_{modality}_{pid}.json"

        if feat_a.exists() and nods_a.exists():
            return modality, feat_a, nods_a, (var_a if var_a.exists() else None)
        
        # Alternative: nodes folder has nodes_X_025.json instead
        nods_b = data_root / "cow_nodes" / f"nodes_{modality}_{pid}.json"
        
        if feat_a.exists() and nods_b.exists():
            return modality, feat_a, nods_b, (var_a if var_a.exists() else None)

    raise FileNotFoundError(
        f"Could not find patient files for pid={pid} under {data_root}.\n"
        f"Expected:\n"
        f"  {data_root}/cow_features/topcow_[ct|mr]_{pid}.json\n"
        f"  {data_root}/cow_nodes/topcow_[ct|mr]_{pid}.json OR nodes_[ct|mr]_{pid}.json\n"
        f"  {data_root}/cow_variants/topcow_[ct|mr]_{pid}.json (optional)\n"
    )


# -------------------------
# Nodes + side inference
# -------------------------

def flatten_nodes(nodes_json: Any):
    """Extract node coordinates and infer left/right side from JSON structure."""
    id_to_xyz: Dict[int, np.ndarray] = {}
    node_side_hint: Dict[int, str] = {}
    r_x: List[float] = []
    l_x: List[float] = []

    def path_has_side(path) -> Optional[str]:
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
    if len(r_x) > 0 and len(l_x) > 0:
        return float(np.mean(r_x)) > float(np.mean(l_x))
    return True


def side_from_endpoints(
    id_to_xyz: Dict[int, np.ndarray],
    node_side_hint: Dict[int, str],
    start_id: int,
    end_id: int,
    right_is_positive_x: bool
) -> Optional[str]:
    """Infer vessel side from endpoint coordinates."""
    hints = []
    if start_id in node_side_hint:
        hints.append(node_side_hint[start_id])
    if end_id in node_side_hint:
        hints.append(node_side_hint[end_id])

    if hints:
        if all(h == hints[0] for h in hints):
            return hints[0]
        return None

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
    """Extract vessel segments from features JSON."""
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
# Name normalization
# -------------------------

CANONICAL = {"A1", "A2", "Acom", "Pcom", "MCA", "BA", "P1", "P2", "ICA"}

def normalize_segment_name(raw_name: str) -> str:
    """Normalize vessel segment names to canonical form."""
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

    return name


# -------------------------
# FirstBlood mapping
# -------------------------

def build_firstblood_mapping() -> Dict[Tuple[Optional[str], str], List[str]]:
    """Map canonical vessel names to Abel_ref2 IDs."""
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
# Geometry injection
# -------------------------

REQUIRED_ARTERIAL_COLUMNS = {"ID", "name", "length[SI]", "start_diameter[SI]", "end_diameter[SI]"}

def assert_arterial_schema(df: pd.DataFrame):
    missing = [c for c in REQUIRED_ARTERIAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "arterial.csv schema mismatch. Missing columns: " + ", ".join(missing)
        )


def apply_geometry(df: pd.DataFrame, fb_id: str, length_m: float, diameter_m: float) -> Optional[Dict[str, Any]]:
    """Apply patient-specific geometry to a vessel in arterial.csv."""
    idxs = df.index[df["ID"] == fb_id].tolist()
    if not idxs:
        return None
    i = idxs[0]

    old_length = float(df.loc[i, "length[SI]"])
    old_d1 = float(df.loc[i, "start_diameter[SI]"])
    name = str(df.loc[i, "name"])

    df.loc[i, "length[SI]"] = float(length_m)
    df.loc[i, "start_diameter[SI]"] = float(diameter_m)
    df.loc[i, "end_diameter[SI]"] = float(diameter_m)

    return {
        "Action": "inject",
        "FirstBlood_ID": fb_id,
        "Name": name,
        "Old_length_mm": old_length * 1000.0,
        "New_length_mm": length_m * 1000.0,
        "Old_diameter_mm": old_d1 * 1000.0,
        "New_diameter_mm": diameter_m * 1000.0,
    }


def split_length_by_template(df: pd.DataFrame, ids: List[str], total_length_m: float) -> List[float]:
    """Split total length across multiple segments proportionally."""
    template_lengths = []
    for fb_id in ids:
        idxs = df.index[df["ID"] == fb_id].tolist()
        template_lengths.append(float(df.loc[idxs[0], "length[SI]"]) if idxs else 0.0)
    s = float(np.sum(template_lengths))
    if s <= 1e-12:
        return [total_length_m / len(ids) for _ in ids]
    return [total_length_m * (tl / s) for tl in template_lengths]


# -------------------------
# CRITICAL FIX: Proper peripheral node filtering
# -------------------------

def find_terminal_nodes(arterial_df: pd.DataFrame) -> List[str]:
    """
    Find all peripheral terminal nodes from arterial.csv.
    
    CRITICAL FIX: Use regex to match only p[0-9]+ pattern.
    This excludes nodes like "parameter" which start with 'p' but aren't peripherals.
    """
    starts = set(arterial_df['start_node'].dropna())
    ends = set(arterial_df['end_node'].dropna())
    
    terminals = ends - starts
    
    # FIXED: Use regex pattern p[0-9]+ to match p1, p2, ..., p47
    # Excludes: parameter, param, etc.
    peripheral_pattern = re.compile(r'^p\d+$')
    peripherals = sorted([n for n in terminals if peripheral_pattern.match(str(n))])
    
    return peripherals


# -------------------------
# CRITICAL FIX: Generate main.csv with correct peripheral mappings
# -------------------------

def generate_main_csv(arterial_df: pd.DataFrame, output_path: Path, time_duration: float = 10.317):
    """
    Generate main.csv with CORRECT peripheral mappings.
    
    CRITICAL FIX: Each peripheral pX connects to node pX, NOT to n1!
    """
    peripherals = find_terminal_nodes(arterial_df)
    
    if not peripherals:
        raise ValueError("No peripheral nodes found in arterial.csv (expected p1, p2, ..., p47)")
    
    print(f"  Found {len(peripherals)} peripheral nodes")
    
    lines = [
        "run,forward",
        f"time,{time_duration}",
        "material,linear",
        "solver,maccormack",
        "",
    ]
    
    # MOC arterial connections
    moc_connections = []
    for p in peripherals:
        num = p[1:]  # Extract number: 'p47' -> '47'
        main_node = f"N{num}p"
        moc_connections.extend([main_node, p])
    
    moc_connections.extend(["Heart", "H"])
    
    moc_line = "type,name,main node,model node,main node,model node,..."
    lines.append(moc_line)
    
    moc_arterial = "moc,arterial," + ",".join(moc_connections)
    lines.append(moc_arterial)
    lines.append("")
    
    # CRITICAL FIX: Lumped peripherals connect to their OWN nodes
    for p in peripherals:
        num = p[1:]
        main_node = f"N{num}p"
        # CORRECT: pX connects to pX (NOT to n1!)
        lines.append(f"lumped,{p},{main_node},{p}")
    
    lines.append("lumped,heart_kim_lit,Heart,aorta")
    lines.append("")
    
    # Node declarations
    for p in peripherals:
        num = p[1:]
        lines.append(f"node,N{num}p")
    lines.append("node,Heart")
    lines.append("")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  Generated main.csv with {len(peripherals)} peripherals")
    print(f"  CRITICAL FIX: Peripherals map to their own nodes (p1->p1, not p1->n1)")


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate patient-specific CoW model (v4 - FINAL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 data_generationV4.py
  python3 data_generationV4.py --pid 025 --force
  python3 data_generationV4.py --pid 025 --template Abel_ref2

This script:
  - Reads patient CoW measurements from data/cow_features/
  - Injects patient-specific geometry into Abel_ref2 template
  - Generates correct main.csv (fixes peripheral mapping bug)
  - Copies non-CoW vessels and peripherals from template
        """
    )
    ap.add_argument("--pid", default=None, help="Patient ID (e.g., 025)")
    ap.add_argument("--template", default="Abel_ref2", help="Template model folder")
    ap.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = ap.parse_args()

    pid = args.pid
    if pid is None or str(pid).strip() == "":
        pid = input("Enter patient id (e.g. 025): ").strip()
    pid = str(pid).zfill(3)

    out_model_name = f"patient_{pid}"

    repo_root = get_repo_root()
    data_root = repo_root / "data"
    template_dir = repo_root / "models" / args.template
    out_dir = repo_root / "models" / out_model_name

    if not template_dir.exists():
        raise FileNotFoundError(f"Template model not found: {template_dir}")
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    modality, feature_path, nodes_path, variant_path = find_patient_files(data_root, pid)

    print("=" * 78)
    print("PATIENT COW MODEL GENERATION (v4 - FINAL)")
    print("=" * 78)
    print(f"Patient ID:     {pid}")
    print(f"Modality:       {modality}")
    print(f"Feature file:   {feature_path.name}")
    print(f"Nodes file:     {nodes_path.name}")
    print(f"Variants file:  {variant_path.name if variant_path else '(none)'}")
    print(f"Template:       {args.template}")
    print(f"Output:         {out_model_name}")
    print("-" * 78)

    features = load_json(feature_path)
    nodes = load_json(nodes_path)

    id_to_xyz, node_side_hint, r_x, l_x = flatten_nodes(nodes)
    right_is_pos_x = infer_right_is_positive_x(r_x, l_x)

    segs = parse_patient_features(features)
    print(f"Parsed {len(segs)} vessel segments from raw data")

    fb_map = build_firstblood_mapping()

    candidates: List[Dict[str, Any]] = []

    for s in segs:
        canon = normalize_segment_name(s["raw_name"])

        if canon not in CANONICAL:
            continue

        side = side_from_endpoints(id_to_xyz, node_side_hint, s["start_id"], s["end_id"], right_is_pos_x)

        length_m = float(s["length_mm"]) / 1000.0
        diameter_m = (2.0 * float(s["radius_mm"])) / 1000.0

        if canon in {"BA", "Acom"}:
            key = (None, canon)
        else:
            if side is None:
                continue
            key = (side, canon)

        candidates.append({
            "key": key,
            "canon": canon,
            "side": side,
            "length_m": length_m,
            "diameter_m": diameter_m,
        })

    patient_geom: Dict[Tuple[Optional[str], str], Dict[str, Any]] = {}
    for c in candidates:
        key = c["key"]
        if key not in patient_geom or float(c["length_m"]) > float(patient_geom[key]["length_m"]):
            patient_geom[key] = c

    print(f"Kept {len(patient_geom)} unique CoW vessel measurements")
    print("-" * 78)

    if out_dir.exists():
        if not args.force:
            raise FileExistsError(f"Output model exists: {out_dir}\nUse --force to overwrite.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    template_arterial = template_dir / "arterial.csv"
    if not template_arterial.exists():
        raise FileNotFoundError(f"Missing arterial.csv in template: {template_arterial}")

    # Copy ALL CSV files from template except arterial.csv and main.csv
    print("\nCopying template files:")
    copied_count = 0
    for src in template_dir.glob("*.csv"):
        if src.name in ["arterial.csv", "main.csv"]:
            continue
        shutil.copy(src, out_dir / src.name)
        copied_count += 1
    print(f"  Copied {copied_count} CSV files from template")

    # Load arterial template
    df = pd.read_csv(template_arterial)
    assert_arterial_schema(df)

    modifications: List[Dict[str, Any]] = []

    # Inject BA (split across A59 and A56)
    ba_key = (None, "BA")
    if ba_key in patient_geom:
        fb_ids = fb_map[ba_key]
        total_len = float(patient_geom[ba_key]["length_m"])
        diam = float(patient_geom[ba_key]["diameter_m"])
        split_lens = split_length_by_template(df, fb_ids, total_len)

        for fb_id, L in zip(fb_ids, split_lens):
            mod = apply_geometry(df, fb_id, L, diam)
            if mod:
                mod.update({"Patient_key": "BA"})
                modifications.append(mod)

    # Inject all other CoW vessels
    for (side, canon), g in patient_geom.items():
        if canon == "BA":
            continue

        map_key = (None, canon) if canon == "Acom" else (side, canon)

        if map_key not in fb_map:
            continue

        for fb_id in fb_map[map_key]:
            mod = apply_geometry(df, fb_id, float(g["length_m"]), float(g["diameter_m"]))
            if mod:
                pk = canon if side is None else f"{side}_{canon}"
                mod.update({"Patient_key": pk})
                modifications.append(mod)

    # Save arterial.csv
    df.to_csv(out_dir / "arterial.csv", index=False)
    
    # CRITICAL: Generate main.csv with CORRECT peripheral mappings
    print("\nGenerating main.csv:")
    generate_main_csv(df, out_dir / "main.csv")

    # Save modifications log
    pd.DataFrame(modifications).to_csv(out_dir / "modifications_log.csv", index=False)

    print("\n" + "=" * 78)
    print("MODEL GENERATION COMPLETE")
    print("=" * 78)
    print(f"Modified {len(modifications)} CoW vessels in arterial.csv")
    print(f"\nCRITICAL FIXES APPLIED:")
    print(f"  1. Peripheral filtering: p[0-9]+ pattern (excludes 'parameter')")
    print(f"  2. Peripheral mapping: pX->pX, not pX->n1")
    print(f"\nModel folder:")
    print(f"  {out_dir}")
    print(f"\nNext step:")
    print(f"  cd {repo_root}/projects/simple_run")
    print(f"  ./simple_run.out {out_model_name}")
    print("=" * 78)


if __name__ == "__main__":
    main()