#!/usr/bin/env python3
"""
Modular CoW geometry injection into a solver-compatible FirstBlood model scaffold.

Design intent
-------------
- Abel_ref2 is used ONLY as a scaffold:
  * provides required CSV schema and systemic network + heart/BC defaults.
- Patient-specific content ("yours"):
  * arterial.csv is modified for CoW vessels
  * optionally main.csv can be overwritten if you have a custom one

Inputs (per patient)
--------------------
data/cow_features/feature_mr_<PATIENT>.json
data/cow_nodes/nodes_mr_<PATIENT>.json        (optional)
data/cow_variants/variant_mr_<PATIENT>.json   (optional)

Run
---
cd ~/first_blood/Pipeline
python3 inject_patient_model.py --patient 025 --out-model patient025_CoW_v2
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

import pandas as pd


# ----------------------------
# Configuration / constants
# ----------------------------

@dataclass
class PatientVessel:
    key: str               # e.g. "BA", "R_MCA"
    length_mm: float
    diameter_mm: float     # derived from radius mean
    length_m: float
    diameter_m: float


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _patient_paths(data_root: Path, patient: str) -> Tuple[Path, Path, Path]:
    feat = data_root / "cow_features" / f"feature_mr_{patient}.json"
    nodes = data_root / "cow_nodes" / f"nodes_mr_{patient}.json"
    var = data_root / "cow_variants" / f"variant_mr_{patient}.json"
    return feat, nodes, var


def _extract_patient_vessels(features: dict) -> Dict[str, PatientVessel]:
    """
    Extract a compact vessel dictionary from your nested feature json.

    Output keys are semantic and stable:
      BA, ACOM,
      R_ICA, L_ICA,
      R_MCA, L_MCA,
      R_ACA, L_ACA,
      R_PCA, L_PCA,
      R_PCOM, L_PCOM,
      R_P1, L_P1,
      R_P2, L_P2,
      R_A1, L_A1,
      R_A2, L_A2

    Note: Some of these will not exist in every patient. We only return what exists.
    """
    # Collect raw items by base name, keep list to resolve later
    collected: Dict[str, List[Tuple[float, float]]] = {}

    for region_id, region_data in features.items():
        if not isinstance(region_data, dict):
            continue
        for vessel_name, vessel_list in region_data.items():
            if not isinstance(vessel_name, str):
                continue
            if "bifurcation" in vessel_name.lower():
                continue
            if not isinstance(vessel_list, list) or len(vessel_list) == 0:
                continue

            info = vessel_list[0]
            if not isinstance(info, dict):
                continue
            if "length" not in info or "radius" not in info:
                continue
            if not isinstance(info["radius"], dict) or "mean" not in info["radius"]:
                continue

            length_mm = float(info["length"])
            radius_mm = float(info["radius"]["mean"])
            diameter_mm = 2.0 * radius_mm

            # Normalize vessel label from your dataset vocabulary
            # Your feature file uses names like: "A1","A2","P1","P2","MCA","BA","Acom","Pcom","ICA","ACA","PCA"
            base = vessel_name.strip()

            collected.setdefault(base, []).append((length_mm, diameter_mm))

    # Resolve left/right for paired vessels:
    # We *cannot* truly infer anatomical L/R from list order alone.
    # But your dataset structure is consistent across patients; for now we keep the legacy behavior:
    # first occurrence -> R, second -> L.
    #
    # If later you want robust L/R, we can use nodes_mr_XXX.json semantic labels to infer side.
    out: Dict[str, PatientVessel] = {}

    def add(key: str, Lmm: float, Dmm: float):
        out[key] = PatientVessel(
            key=key,
            length_mm=Lmm,
            diameter_mm=Dmm,
            length_m=Lmm / 1000.0,
            diameter_m=Dmm / 1000.0,
        )

    for base, arr in collected.items():
        base_upper = base.upper()

        if base_upper in ["BA", "ACOM"]:
            # single midline vessel
            Lmm, Dmm = arr[0]
            add(base_upper, Lmm, Dmm)
            continue

        # paired vessels
        if len(arr) >= 1:
            Lmm, Dmm = arr[0]
            add(f"R_{base_upper}", Lmm, Dmm)
        if len(arr) >= 2:
            Lmm, Dmm = arr[1]
            add(f"L_{base_upper}", Lmm, Dmm)

    return out


def _copy_scaffold_model(scaffold_dir: Path, out_model_dir: Path):
    out_model_dir.mkdir(parents=True, exist_ok=True)
    for fp in scaffold_dir.glob("*.csv"):
        shutil.copy(fp, out_model_dir / fp.name)


def _maybe_override_main(out_model_dir: Path, main_override: Optional[Path]):
    if main_override is None:
        return
    if not main_override.exists():
        raise FileNotFoundError(f"--main-override not found: {main_override}")
    shutil.copy(main_override, out_model_dir / "main.csv")


def _update_row_geometry(df: pd.DataFrame, idx: int, new_len_m: float, new_d_m: float):
    # Minimal consistent edits: length and diameters, thickness as 10% of diameter
    df.loc[idx, "length[SI]"] = float(new_len_m)
    df.loc[idx, "start_diameter[SI]"] = float(new_d_m)
    df.loc[idx, "end_diameter[SI]"] = float(new_d_m)
    df.loc[idx, "start_thickness[SI]"] = float(new_d_m) * 0.1
    df.loc[idx, "end_thickness[SI]"] = float(new_d_m) * 0.1


def _ratio_split(total: float, a: float, b: float) -> Tuple[float, float]:
    s = a + b
    if s <= 0:
        return total * 0.5, total * 0.5
    return total * (a / s), total * (b / s)


def inject_patient(
    patient: str,
    data_root: Path,
    scaffold_model: str,
    models_root: Path,
    out_model: str,
    main_override: Optional[Path] = None,
) -> Path:
    feat_fp, nodes_fp, var_fp = _patient_paths(data_root, patient)
    features = _load_json(feat_fp)
    nodes = _load_json(nodes_fp)   # optional
    variant = _load_json(var_fp)   # optional

    if features is None:
        raise FileNotFoundError(f"Missing features: {feat_fp}")

    patient_vessels = _extract_patient_vessels(features)

    scaffold_dir = models_root / scaffold_model
    out_model_dir = models_root / out_model

    if not scaffold_dir.exists():
        raise FileNotFoundError(f"Scaffold model folder not found: {scaffold_dir}")

    print("=" * 80)
    print(f"Injecting patient {patient} into scaffold {scaffold_model}")
    print(f"Data root:  {data_root}")
    print(f"Out model:  {out_model_dir}")
    print("=" * 80)

    # 1) Copy scaffold (schema + heart/systemic defaults)
    _copy_scaffold_model(scaffold_dir, out_model_dir)

    # 2) Optionally override main.csv
    _maybe_override_main(out_model_dir, main_override)

    arterial_fp = out_model_dir / "arterial.csv"
    df = pd.read_csv(arterial_fp)

    # IMPORTANT:
    # This mapping is to the scaffold topology (Abel_ref2 IDs).
    # Because you reuse Abel_ref2 topology, this stays constant across patients.
    #
    # Basilar special-case: patient has BA; scaffold has 2 segments (A59 + A56).
    FB_MAP = {
        "R_ICA": "A12",
        "L_ICA": "A16",
        "R_MCA": "A70",
        "L_MCA": "A73",
        "R_PCOM": "A62",
        "L_PCOM": "A63",
        "ACOM": "A77",
        # PCA segments in Abel_ref2: P1 (A60/A61) and P2 (A64/A65)
        "R_P1": "A60",
        "L_P1": "A61",
        "R_P2": "A64",
        "L_P2": "A65",
        # ACA proximal/distal in scaffold
        "R_A2": "A76",
        "L_A2": "A78",
        # If your patient file uses ACA without A1/A2 split, you can still update A2
        # and leave proximal ACA segments unchanged.
    }

    modifications = []

    def log_change(patient_key: str, fb_id: str, old_len_m: float, old_d_m: float, new_len_m: float, new_d_m: float, fb_name: str):
        modifications.append({
            "patient": patient,
            "patient_key": patient_key,
            "fb_id": fb_id,
            "fb_name": fb_name,
            "old_length_mm": old_len_m * 1000.0,
            "new_length_mm": new_len_m * 1000.0,
            "old_diameter_mm": old_d_m * 1000.0,
            "new_diameter_mm": new_d_m * 1000.0,
            "length_change_%": ((new_len_m - old_len_m) / old_len_m * 100.0) if old_len_m > 0 else None,
            "diameter_change_%": ((new_d_m - old_d_m) / old_d_m * 100.0) if old_d_m > 0 else None,
        })

    # 3) Basilar split (recommended)
    # Scaffold basilar pieces:
    BA1_ID = "A59"
    BA2_ID = "A56"
    if "BA" in patient_vessels:
        pv = patient_vessels["BA"]

        # Baseline lengths (used only for split ratio; not for final total)
        row1 = df[df["ID"] == BA1_ID]
        row2 = df[df["ID"] == BA2_ID]
        if len(row1) == 1 and len(row2) == 1:
            i1 = int(row1.index[0])
            i2 = int(row2.index[0])

            old1_len = float(df.loc[i1, "length[SI]"])
            old2_len = float(df.loc[i2, "length[SI]"])
            new1_len, new2_len = _ratio_split(pv.length_m, old1_len, old2_len)

            old1_d = float(df.loc[i1, "start_diameter[SI]"])
            old2_d = float(df.loc[i2, "start_diameter[SI]"])

            _update_row_geometry(df, i1, new1_len, pv.diameter_m)
            _update_row_geometry(df, i2, new2_len, pv.diameter_m)

            log_change("BA->BA1", BA1_ID, old1_len, old1_d, new1_len, pv.diameter_m, str(df.loc[i1, "name"]))
            log_change("BA->BA2", BA2_ID, old2_len, old2_d, new2_len, pv.diameter_m, str(df.loc[i2, "name"]))

            print(f"✓ BA split across {BA1_ID}+{BA2_ID}: total {pv.length_mm:.2f} mm")
        else:
            print("⚠ Could not locate both BA scaffold segments (A59 and A56). Skipping BA update.")

    # 4) Standard vessel updates
    for pkey, fb_id in FB_MAP.items():
        if pkey not in patient_vessels:
            continue
        pv = patient_vessels[pkey]
        row = df[df["ID"] == fb_id]
        if len(row) != 1:
            print(f"⚠ Could not locate scaffold vessel {fb_id} for {pkey}. Skipping.")
            continue
        idx = int(row.index[0])

        old_len = float(df.loc[idx, "length[SI]"])
        old_d = float(df.loc[idx, "start_diameter[SI]"])
        fb_name = str(df.loc[idx, "name"])

        _update_row_geometry(df, idx, pv.length_m, pv.diameter_m)
        log_change(pkey, fb_id, old_len, old_d, pv.length_m, pv.diameter_m, fb_name)
        print(f"✓ {pkey:7s} -> {fb_id}: L {old_len*1000:6.2f}->{pv.length_mm:6.2f} mm | D {old_d*1000:5.2f}->{pv.diameter_mm:5.2f} mm")

    # 5) Save
    df.to_csv(arterial_fp, index=False)

    # 6) Log
    log_fp = out_model_dir / "modifications_log.csv"
    if modifications:
        pd.DataFrame(modifications).to_csv(log_fp, index=False)
        print(f"✓ Wrote modification log: {log_fp}")

    # Optional: print variant summary (no geometry changes done here)
    if variant is not None:
        print("\nVariant file loaded (not applied automatically):")
        print(variant)

    # nodes available for future L/R inference improvements
    if nodes is not None:
        print("\nNodes file loaded (available for future robust L/R inference).")

    print("\nNext step:")
    print("  cd ~/first_blood/projects/simple_run")
    print(f"  ./simple_run.out {out_model}")
    return out_model_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", required=True, help="patient id like 025")
    ap.add_argument("--data-root", default=str(Path.home() / "first_blood/Pipeline/data"),
                    help="Pipeline data folder containing cow_features/, cow_nodes/, cow_variants/")
    ap.add_argument("--scaffold-model", default="Abel_ref2",
                    help="solver-compatible scaffold model (default: Abel_ref2)")
    ap.add_argument("--models-root", default=str(Path.home() / "first_blood/models"),
                    help="FirstBlood models folder")
    ap.add_argument("--out-model", required=True,
                    help="name of new model folder to create in models-root")
    ap.add_argument("--main-override", default=None,
                    help="optional path to your own main.csv to overwrite the scaffold one")
    args = ap.parse_args()

    inject_patient(
        patient=args.patient,
        data_root=Path(args.data_root),
        scaffold_model=args.scaffold_model,
        models_root=Path(args.models_root),
        out_model=args.out_model,
        main_override=Path(args.main_override) if args.main_override else None,
    )


if __name__ == "__main__":
    main()
