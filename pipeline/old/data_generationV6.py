#!/usr/bin/env python3
"""
Patient-specific Circle of Willis model generator (v6 - PRODUCTION FINAL)

CORRECT APPROACH (matches inject_patient_cow_CORRECT.py):
- Copy ALL template files INCLUDING main.csv (DO NOT generate new main.csv)
- Only modify arterial.csv with patient-specific CoW geometry
- Preserve all node declarations exactly as in Abel_ref2

This is the working approach used by inject_patient_cow_CORRECT.py

Usage:
  python3 data_generationV6.py
  OR
  python3 data_generationV6.py --pid 025 --force
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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
    """Find patient data files."""
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

    raise FileNotFoundError(f"Could not find patient files for pid={pid}")


def flatten_nodes(nodes_json: Any):
    """Extract node coordinates and infer sides."""
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
    """Infer vessel side."""
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
    """Extract vessel segments."""
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


def apply_geometry(df: pd.DataFrame, fb_id: str, length_m: float, diameter_m: float) -> Optional[Dict[str, Any]]:
    """Apply patient geometry to vessel."""
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
    template_lengths = []
    for fb_id in ids:
        idxs = df.index[df["ID"] == fb_id].tolist()
        template_lengths.append(float(df.loc[idxs[0], "length[SI]"]) if idxs else 0.0)
    s = float(np.sum(template_lengths))
    if s <= 1e-12:
        return [total_length_m / len(ids) for _ in ids]
    return [total_length_m * (tl / s) for tl in template_lengths]


def main():
    ap = argparse.ArgumentParser(
        description="Patient-specific CoW model generator (v6 - PRODUCTION FINAL)"
    )
    ap.add_argument("--pid", default=None, help="Patient ID")
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
    print("PATIENT-SPECIFIC CoW MODEL GENERATOR (v6 - FINAL)")
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
    print(f"Parsed {len(segs)} vessel segments")

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

    print(f"Kept {len(patient_geom)} unique CoW measurements")
    print("-" * 78)

    if out_dir.exists():
        if not args.force:
            raise FileExistsError(f"Output exists: {out_dir}\nUse --force")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    template_arterial = template_dir / "arterial.csv"
    if not template_arterial.exists():
        raise FileNotFoundError(f"Missing arterial.csv in template")

    # CRITICAL: Copy ALL template CSV files (including main.csv)
    # DO NOT generate new main.csv!
    print("\nCopying template files:")
    copied = 0
    for src in template_dir.glob("*.csv"):
        shutil.copy(src, out_dir / src.name)
        copied += 1
    print(f"  Copied {copied} CSV files (including main.csv)")

    # Load arterial for modification
    arterial_path = out_dir / "arterial.csv"
    df = pd.read_csv(arterial_path)

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

    # Save modified arterial.csv
    df.to_csv(arterial_path, index=False)
    print(f"\nModified arterial.csv:")
    print(f"  {len(modifications)} CoW vessels updated")

    # Save log
    pd.DataFrame(modifications).to_csv(out_dir / "modifications_log.csv", index=False)

    print("\n" + "=" * 78)
    print("SUCCESS")
    print("=" * 78)
    print(f"Output: {out_dir}")
    print(f"\nCRITICAL: main.csv copied directly from Abel_ref2")
    print(f"          (NOT generated - preserves all node declarations)")
    print(f"\nRun simulation:")
    print(f"  cd {repo_root}/projects/simple_run")
    print(f"  ./simple_run.out {out_model_name}")
    print("=" * 78)


if __name__ == "__main__":
    main()