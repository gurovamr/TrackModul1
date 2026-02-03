#!/usr/bin/env python3
"""
Patient-specific Circle of Willis model generator for first_blood (v5 - PRODUCTION)

COMPLETE FIX - ALL ISSUES RESOLVED:
1. Peripheral node filtering: regex p[0-9]+ (excludes "parameter")
2. Correct main.csv: pX -> pX mappings (not pX -> n1)
3. Complete arterial.csv: includes ALL node declarations (n nodes + p nodes)

This is the production-ready version for generating patient-specific models.

Run from: <repo_root>/pipeline

Usage:
  python3 data_generationV5.py
  (will prompt for patient ID)

OR:
  python3 data_generationV5.py --pid 025 --force

Reads patient raw data from:
  <repo_root>/data/
    cow_features/topcow_[ct|mr]_<pid>.json
    cow_nodes/topcow_[ct|mr]_<pid>.json
    cow_variants/topcow_[ct|mr]_<pid>.json (optional)

Creates:
  <repo_root>/models/patient_<pid>/
    - arterial.csv (patient CoW geometry + ALL node declarations)
    - main.csv (correct peripheral mappings)
    - p1.csv, ..., p47.csv (from template)
    - heart_kim_lit.csv (from template)
    - All other template CSV files
"""

import argparse
import json
import shutil
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
import pandas as pd


def get_repo_root() -> Path:
    pipeline_dir = Path(__file__).resolve().parent
    return pipeline_dir.parent


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


def find_patient_files(data_root: Path, pid: str) -> Tuple[str, Path, Path, Optional[Path]]:
    """Find patient data files with flexible naming."""
    pid = str(pid).zfill(3)

    for modality in ["ct", "mr"]:
        feat_a = data_root / "cow_features" / f"topcow_{modality}_{pid}.json"
        nods_a = data_root / "cow_nodes" / f"topcow_{modality}_{pid}.json"
        var_a  = data_root / "cow_variants" / f"topcow_{modality}_{pid}.json"

        if feat_a.exists() and nods_a.exists():
            return modality, feat_a, nods_a, (var_a if var_a.exists() else None)
        
        nods_b = data_root / "cow_nodes" / f"nodes_{modality}_{pid}.json"
        if feat_a.exists() and nods_b.exists():
            return modality, feat_a, nods_b, (var_a if var_a.exists() else None)

    raise FileNotFoundError(
        f"Could not find patient files for pid={pid} under {data_root}.\n"
        f"Expected:\n"
        f"  {data_root}/cow_features/topcow_[ct|mr]_{pid}.json\n"
        f"  {data_root}/cow_nodes/topcow_[ct|mr]_{pid}.json OR nodes_[ct|mr]_{pid}.json\n"
    )


def flatten_nodes(nodes_json: Any):
    """Extract node coordinates and infer left/right side."""
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

        segs.append({
            "raw_name": raw_name.strip(),
            "start_id": start_id,
            "end_id": end_id,
            "length_mm": length_mm,
            "radius_mm": radius_mm,
        })

    _walk_json(features_json, on_dict=on_dict)
    return segs


CANONICAL = {"A1", "A2", "Acom", "Pcom", "MCA", "BA", "P1", "P2", "ICA"}

def normalize_segment_name(raw_name: str) -> str:
    """Normalize vessel names to canonical form."""
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


REQUIRED_ARTERIAL_COLUMNS = {"ID", "name", "length[SI]", "start_diameter[SI]", "end_diameter[SI]"}

def assert_arterial_schema(df: pd.DataFrame):
    missing = [c for c in REQUIRED_ARTERIAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("arterial.csv missing columns: " + ", ".join(missing))


def apply_geometry(df: pd.DataFrame, fb_id: str, length_m: float, diameter_m: float) -> Optional[Dict[str, Any]]:
    """Apply patient-specific geometry to a vessel."""
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
    """Split length across multiple segments proportionally."""
    template_lengths = []
    for fb_id in ids:
        idxs = df.index[df["ID"] == fb_id].tolist()
        template_lengths.append(float(df.loc[idxs[0], "length[SI]"]) if idxs else 0.0)
    s = float(np.sum(template_lengths))
    if s <= 1e-12:
        return [total_length_m / len(ids) for _ in ids]
    return [total_length_m * (tl / s) for tl in template_lengths]


def extract_all_nodes_from_arterial(df: pd.DataFrame) -> Set[str]:
    """Extract all unique node names from arterial vessels."""
    nodes = set()
    for col in ['start_node', 'end_node']:
        if col in df.columns:
            nodes.update(df[col].dropna().astype(str).unique())
    return nodes


def find_terminal_nodes(arterial_df: pd.DataFrame) -> List[str]:
    """
    Find peripheral terminal nodes (p1, p2, ..., p47).
    Uses regex to exclude nodes like "parameter".
    """
    starts = set(arterial_df['start_node'].dropna())
    ends = set(arterial_df['end_node'].dropna())
    terminals = ends - starts
    peripheral_pattern = re.compile(r'^p\d+$')
    peripherals = sorted([n for n in terminals if peripheral_pattern.match(str(n))])
    return peripherals


def add_node_declarations_to_arterial(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add node declarations to arterial.csv for all nodes.
    
    CRITICAL: The solver requires ALL nodes to be declared in arterial.csv,
    including peripheral terminal nodes (p1, p2, ...).
    """
    # Extract all unique nodes from vessel endpoints
    all_nodes = extract_all_nodes_from_arterial(df)
    
    # Find which nodes already have declarations
    existing_nodes = set()
    node_rows = df[df['type'] == 'node']
    if len(node_rows) > 0 and 'ID' in df.columns:
        existing_nodes = set(node_rows['ID'].dropna().astype(str))
    
    # Nodes that need to be added
    missing_nodes = all_nodes - existing_nodes
    
    if not missing_nodes:
        return df
    
    # Create new node declaration rows
    new_rows = []
    for node in sorted(missing_nodes):
        # Node declaration format: type,ID,name,compliance,resistance,...
        # Use same format as template nodes
        new_row = {
            'type': 'node',
            'ID': node,
            'name': 0,  # Compliance
            'start_node': 4.27E+10,  # Resistance (high for terminals)
        }
        new_rows.append(new_row)
    
    # Append new node declarations
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
    
    return df


def generate_main_csv(arterial_df: pd.DataFrame, output_path: Path, time_duration: float = 10.317):
    """Generate main.csv with correct peripheral mappings."""
    peripherals = find_terminal_nodes(arterial_df)
    
    if not peripherals:
        raise ValueError("No peripheral nodes (p1, p2, ...) found in arterial.csv")
    
    print(f"  Found {len(peripherals)} peripheral nodes")
    
    lines = [
        "run,forward",
        f"time,{time_duration}",
        "material,linear",
        "solver,maccormack",
        "",
        "type,name,main node,model node,main node,model node,...",
    ]
    
    # MOC arterial connections
    moc_connections = []
    for p in peripherals:
        num = p[1:]
        main_node = f"N{num}p"
        moc_connections.extend([main_node, p])
    moc_connections.extend(["Heart", "H"])
    
    lines.append("moc,arterial," + ",".join(moc_connections))
    lines.append("")
    
    # Lumped peripherals - CORRECT mappings
    for p in peripherals:
        num = p[1:]
        main_node = f"N{num}p"
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
    
    print(f"  Generated main.csv: {len(peripherals)} peripherals (pX->pX)")


def main():
    ap = argparse.ArgumentParser(
        description="Generate patient-specific CoW model (v5 - PRODUCTION)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Production-ready patient-specific model generator.

Generates:
  - arterial.csv with patient CoW geometry + ALL node declarations
  - main.csv with correct peripheral mappings (pX->pX)
  - All supporting files from template

Example:
  python3 data_generationV5.py --pid 025 --force
        """
    )
    ap.add_argument("--pid", default=None, help="Patient ID (e.g., 025)")
    ap.add_argument("--template", default="Abel_ref2", help="Template model")
    ap.add_argument("--force", action="store_true", help="Overwrite existing")
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
        raise FileNotFoundError(f"Template not found: {template_dir}")
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    modality, feature_path, nodes_path, variant_path = find_patient_files(data_root, pid)

    print("=" * 78)
    print("PATIENT-SPECIFIC MODEL GENERATION (v5 - PRODUCTION)")
    print("=" * 78)
    print(f"Patient ID:     {pid}")
    print(f"Modality:       {modality}")
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
            raise FileExistsError(f"Output exists: {out_dir}\nUse --force to overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    template_arterial = template_dir / "arterial.csv"
    if not template_arterial.exists():
        raise FileNotFoundError(f"Missing arterial.csv in template")

    # Copy ALL template CSV files except arterial.csv and main.csv
    print("\nCopying template files:")
    copied = 0
    for src in template_dir.glob("*.csv"):
        if src.name in ["arterial.csv", "main.csv"]:
            continue
        shutil.copy(src, out_dir / src.name)
        copied += 1
    print(f"  Copied {copied} CSV files from template")

    # Load and modify arterial template
    df = pd.read_csv(template_arterial)
    assert_arterial_schema(df)

    modifications: List[Dict[str, Any]] = []

    # Inject BA
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

    # Inject other CoW vessels
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

    # CRITICAL: Add ALL node declarations to arterial.csv
    print("\nAdding node declarations to arterial.csv:")
    df = add_node_declarations_to_arterial(df)
    all_nodes = extract_all_nodes_from_arterial(df)
    print(f"  Total nodes in network: {len(all_nodes)}")
    
    # Save arterial.csv
    df.to_csv(out_dir / "arterial.csv", index=False)
    print(f"  Saved arterial.csv with complete node declarations")

    # Generate main.csv
    print("\nGenerating main.csv:")
    generate_main_csv(df, out_dir / "main.csv")

    # Save log
    pd.DataFrame(modifications).to_csv(out_dir / "modifications_log.csv", index=False)

    print("\n" + "=" * 78)
    print("MODEL GENERATION COMPLETE")
    print("=" * 78)
    print(f"Modified {len(modifications)} CoW vessels")
    print(f"\nAll issues fixed:")
    print(f"  1. Peripheral filtering: p[0-9]+ regex")
    print(f"  2. Peripheral mapping: pX->pX (not pX->n1)")
    print(f"  3. Node declarations: ALL nodes in arterial.csv")
    print(f"\nOutput: {out_dir}")
    print(f"\nRun simulation:")
    print(f"  cd {repo_root}/projects/simple_run")
    print(f"  ./simple_run.out {out_model_name}")
    print("=" * 78)


if __name__ == "__main__":
    main()