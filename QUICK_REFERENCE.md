# Quick Reference: Patient-Specific Validation

## One-Command Validation

```bash
cd ~/first_blood
python3 pipeline/validate_complete.py --model patient_025 --pid 025 --compare Abel_ref2
```

## What It Checks

| Check | Paper? | Your Job? | Pass Criteria |
|-------|--------|-----------|--------------|
| Cardiac output 4–7 L/min | ✓ | | Auto |
| Aortic pressure waveform | ✓ | | Auto |
| Numerical stability | ✓ | | Auto |
| **Flow asymmetry <30%** | | ✓ | R-ICA ≈ L-ICA |
| **Collateral direction** | | ✓ | Acom/Pcom physiological |
| **Pressure gradients** | | ✓ | Consistent across model |
| **Variant handling** | | ✓ | Absent vessels d<0.5mm |
| **Reference comparison** | | ✓ | ±20% of Abel_ref2 |

## Interpretation

### ✓ All Checks Pass
→ Model is **valid for CoW analysis**  
→ Proceed to: `analyze_cow_flows.py`, `compare_abel.py`, etc.

### ⚠️ Global Validation Has Warnings
→ **Expected** — paper only validates global hemodynamics  
→ **OK to proceed** if patient-specific checks pass  
→ Local absolute pressures are model-dependent

### ✗ Patient-Specific Check Fails
→ **Stop** — fix the issue  
→ See PATIENT_VALIDATION_GUIDE.md for debugging  
→ Common issues: L/R side mismatch, variant not applied, vessel diameter wrong

## Key Results for patient_025

```
Flow Symmetry:           14.0% ✓ (R-ICA 311 / L-ICA 270 mL/min)
Collateral Direction:    ✓ (Acom 12 R→L, Pcoms vertebral→posterior)
Pressure Differences:    ✓ (Consistent -69 mmHg across vessels)
Variant Handling:        ✓ (All vessels present, d=1.2–4.35mm)
Reference Comparison:    ✓ (+18.9% vs Abel_ref2)

STATUS: VALID FOR ANALYSIS ✓
```

## Step-by-Step for New Patient

1. Generate: `python3 pipeline/data_generationV7.py --pid XXX`
2. Simulate: `cd projects/simple_run && ./simple_run.out patient_XXX`
3. Validate: `python3 pipeline/validate_complete.py --model patient_XXX --pid XXX`
4. Analyze: `python3 analysis_V3/analyze_cow_flows.py` (if validation passes)

## Files to Read

| File | When to Read |
|------|--------------|
| VALIDATION_FRAMEWORK.md | Understand overall validation approach |
| PATIENT_VALIDATION_GUIDE.md | Interpret specific check results |
| SIMULATION_ARCHITECTURE.md | Understand vessel topology and IDs |
| pipeline/validate_complete.py | See what checks are run |

## Paper Scope

The FirstBlood paper validates:
- ✓ Global hemodynamics (CO, aortic pressure, stability)

The paper does NOT validate:
- ❌ Local pressures in Circle of Willis
- ❌ Patient-specific anatomy variants
- ❌ Collateral flow realism
- ❌ Pressure differences

**Your responsibility:** Check these patient-specific features yourself.
