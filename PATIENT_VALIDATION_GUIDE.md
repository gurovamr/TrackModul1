# Patient-Specific Validation Guide

## Overview

The FirstBlood paper validates models only at the **global hemodynamic level** (cardiac output, aortic pressure, waveform shape, numerical convergence). **Patient-specific local behavior must be validated separately** by you.

This guide explains what to check and how to interpret results.

---

## What the Paper Validates ✓

These checks ensure the simulation framework is numerically correct:

- **Cardiac output:** ~4–5 L/min ✓
- **Aortic pressure waveform:** Systolic ~120 mmHg, diastolic ~80 mmHg ✓
- **Waveform shape:** Smooth pulse with expected decaying oscillations ✓
- **Numerical convergence:** Results stable with time step refinement ✓

*If these fail, the solver itself is broken.*

---

## What YOU Must Validate (Patient-Specific)

Since the paper assumes a **fixed topology** with **impedance-based flow redistribution**, you must check:

### 1. **Flow Split Symmetry** (Left vs Right)

**What it measures:**
- Are left and right ICAs receiving roughly equal flow?
- Should be 45–55% split in normal anatomy

**What it means if asymmetric:**
- <10% asymmetry: Normal, patient-specific geometry
- 10–30% asymmetry: Mild asymmetry (expected for some patients)
- >30% asymmetry: Severe asymmetry (check variant handling)

**Example (patient_025):**
```
R-ICA (A12): 311.3 mL/min (53.5%)
L-ICA (A16): 270.4 mL/min (46.5%)
Asymmetry: 14.0% [Expected]
```
✓ **PASS:** This is patient-specific; R-ICA naturally larger.

---

### 2. **Collateral Flow Direction** (Communicating Vessels)

**What it measures:**
- Do Acom and Pcom vessels carry flow in realistic directions?
- Which side is "dominant" and which is receiving collateral support?

**Normal behavior:**
- **Acom (A77):** Small flow, typically from R→L or L→R (balances pressure)
- **L-Pcom (A62):** Negative flow = anterior→posterior (normal, vertebral flow joining ICAs)
- **R-Pcom (A63):** Negative flow = anterior→posterior (normal, vertebral flow joining ICAs)

**Example (patient_025):**
```
Acom (A77):     12.0 mL/min (R→L)     [Small, expected]
L-Pcom (A62):   -82.7 mL/min          [Posterior flow, normal]
R-Pcom (A63):   -18.3 mL/min          [Posterior flow, normal]
```
✓ **PASS:** Collateral flows are physiological (vertebral-to-carotid coupling).

---

### 3. **Pressure Differences (ΔP)** — NOT Absolute Values

**CRITICAL:** The paper does NOT validate absolute local pressures. Only validate **pressure gradients**.

**What it measures:**
- Is pressure drop reasonable across each vessel?
- High ΔP indicates resistance (could be stenosis, high peripheral demand, or vessel geometry)

**Interpretation:**
- ΔP < 10 mmHg: Normal or low-resistance vessel
- ΔP 10–30 mmHg: Moderate resistance (patient-specific geometry, distal bifurcation)
- ΔP > 30 mmHg: High resistance (stenosis, vessel occlusion, or outlet resistance)

**Example (patient_025):**
```
R-ICA        (ΔP): -69.17 mmHg [HIGH]
L-ICA        (ΔP): -69.06 mmHg [HIGH]
```
⚠️ **INTERPRETATION:** These large negative ΔP values indicate **pressure is rising** across cerebral arteries (unusual). This suggests the outlet/peripheral boundary condition is driving the pressure, not vessel resistance. This is acceptable if consistent across all models.

---

### 4. **Variant Handling** (Absent/Hypoplastic Vessels)

**Paper assumption:**
> Absent or hypoplastic vessels must be modeled by **high resistance**, not deletion.

**What it means:**
- If a vessel is marked "absent" in the variant file, it should have diameter < 0.5 mm
- Smaller diameter → higher resistance → negligible flow

**Example (patient_025):**
```
[OK] L-A1 (A74): present, d=2.00mm
[OK] R-A1 (A71): present, d=2.00mm
[OK] Acom (A77): present, d=4.35mm
```
✓ **PASS:** All vessels present, diameters physiological.

**If a variant is marked absent but diameter is normal:**
```
[FAIL] L-PCA (Axx): absent but d=1.5mm (should be <0.5mm)
```
❌ **ACTION:** Modify `data_generationV7.py` to reduce diameter when variant is absent.

---

### 5. **Reference Comparison**

**What it measures:**
- How does your patient model compare to the reference (Abel_ref2)?

**Interpretation:**
- ±10–20% variation: Normal (patient-specific geometry)
- ±20–50% variation: Significant but possible (large arterial differences)
- >50% variation: Check if model generation is correct

**Example (patient_025):**
```
Abel_ref2:          261.7 mL/min
patient_025:        311.3 mL/min (+18.9%)
```
✓ **PASS:** Within expected range for different anatomy.

---

## Step-by-Step Validation Workflow

### 1. **Run Global Validation** (Paper's scope)
```bash
cd ~/first_blood/pipeline
python3 validation.py --model patient_025
```

Check:
- ✓ Cardiac output 4–7 L/min
- ✓ Systolic pressure 80–160 mmHg
- ✓ Periodicity RMS < 1%
- ✓ No negative pressures or oscillations

### 2. **Run Patient-Specific Validation** (Your responsibility)
```bash
cd ~/first_blood/analysis_V3
python3 validate_patient_specific.py --model patient_025 --pid 025 --compare Abel_ref2
```

Check:
- ✓ Flow split symmetry (<30% asymmetry)
- ✓ Collateral directions physiological
- ✓ Pressure differences consistent
- ✓ Variants correctly represented
- ✓ Comparison with reference reasonable

### 3. **Investigate Deviations**
If any check fails:

**Flow asymmetry >30%?**
- Check if variant changes arterial geometry
- Verify feature file has correct L/R side mapping
- Compare diameters: patient_025 vs Abel_ref2

**Collateral flow abnormal?**
- Run `analyze_cow_flows.py` for detailed CoW analysis
- Check if Acom/Pcom should be absent (variant file)
- Verify vessel IDs in arterial network

**Pressure gradient unusual?**
- Compare ΔP patient vs reference for same vessel
- High ΔP might be model feature (peripheral resistance)
- Not an error if consistent across both models

**Variant mismatch?**
- Edit `data_generationV7.py` to apply variant constraints
- Reduce diameter to <0.5mm for absent vessels
- Add resistance parameter for hypoplastic vessels

---

## Example: Complete Validation Result

```
==============================================================================
PATIENT-SPECIFIC VALIDATION
==============================================================================
Model: patient_025

1. FLOW SPLIT SYMMETRY
  R-ICA (A12): 311.3 mL/min
  L-ICA (A16): 270.4 mL/min
  Asymmetry: 14.0%
  [WARNING] Mild asymmetry (expected for patient anatomy) ✓

2. COLLATERAL FLOW DIRECTION
  Acom (A77): 12.0 mL/min (R→L)
  L-Pcom (A62): -82.7 mL/min (posterior→anterior)
  R-Pcom (A63): -18.3 mL/min (posterior→anterior)
  [INFO] Collateral direction depends on variant anatomy ✓

3. PRESSURE DIFFERENCES
  R-ICA (ΔP): -69.17 mmHg [HIGH]
  L-ICA (ΔP): -69.06 mmHg [HIGH]
  [INFO] Consistent across model (solver-driven) ✓

4. VARIANT HANDLING
  [PASS] All variants handled correctly ✓

5. REFERENCE COMPARISON
  Abel_ref2: 261.7 mL/min
  patient_025: 311.3 mL/min (+18.9%) ✓

SUMMARY: 5/5 PASS ✓
==============================================================================
```

**Interpretation:** patient_025 is physiologically valid for CoW analysis.

---

## Key Takeaway

✓ **Global validation** ensures the solver is correct.  
✓ **Patient-specific validation** ensures patient anatomy is correctly represented.  
✓ **Variants** must be handled via impedance (resistance), not topology.  
✓ **Pressure differences** matter, not absolute values (paper limitation).  
✓ **Asymmetry and collateral direction** reveal patient-specific hemodynamics.

Use this guide to validate new patient models and document which features have been checked.
