#!/usr/bin/env python3
"""
Biological Assessment for FirstBlood Simulations
=================================================
Evaluates biological correctness of simulation results.
Reads validation.py and analysis.py outputs and provides interpretation.

Usage:
    python3 biological_assessment.py                    # Interactive prompt
    python3 biological_assessment.py --model patient_025
    python3 biological_assessment.py --models patient_025,Abel_ref2
    python3 biological_assessment.py --all
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Optional


def get_repo_root():
    """Get repository root directory"""
    return Path(__file__).resolve().parent.parent


class BiologicalAssessor:
    """Assess biological correctness of simulation results"""
    
    def __init__(self, model_name: str, output_base: Path):
        self.model_name = model_name
        self.validation_file = output_base / "validation" / f"{model_name}_validation.json"
        self.analysis_file = output_base / "analysis" / model_name / "analysis_summary.json"
        
        self.validation_data = None
        self.analysis_data = None
        self.issues = []
        self.warnings = []
        self.passes = []
        
    def load_data(self) -> bool:
        """Load validation and analysis data"""
        if not self.validation_file.exists():
            print(f"[ERROR] Validation data not found: {self.validation_file}")
            print(f"        Run: python3 validation.py --model {self.model_name}")
            return False
        
        if not self.analysis_file.exists():
            print(f"[ERROR] Analysis data not found: {self.analysis_file}")
            print(f"        Run: python3 analysis.py --model {self.model_name}")
            return False
        
        with open(self.validation_file, 'r') as f:
            self.validation_data = json.load(f)
        
        with open(self.analysis_file, 'r') as f:
            self.analysis_data = json.load(f)
        
        return True
    
    def assess_global_hemodynamics(self):
        """Assess global circulation metrics"""
        print("\n" + "="*70)
        print("1. GLOBAL HEMODYNAMICS")
        print("="*70)
        
        # Cardiac output
        co = self.validation_data['validation_results']['cardiac_output']['value_l_min']
        print(f"\nCardiac Output:      {co:.2f} L/min")
        print(f"  Normal range:      4.0 - 7.0 L/min")
        if 4.0 <= co <= 7.0:
            print(f"  Status:            ✓ NORMAL")
            self.passes.append("Cardiac output normal")
        elif 3.5 <= co < 4.0 or 7.0 < co <= 8.0:
            print(f"  Status:            ⚠ BORDERLINE")
            self.warnings.append(f"Cardiac output borderline: {co:.2f} L/min")
        else:
            print(f"  Status:            ✗ ABNORMAL")
            self.issues.append(f"Cardiac output abnormal: {co:.2f} L/min")
        
        # Aortic pressures
        waveform = self.analysis_data['waveform_characteristics']
        systolic = waveform['systolic_mmHg']
        diastolic = waveform['diastolic_mmHg']
        mean_p = waveform['mean_pressure_mmHg']
        pulse_p = waveform['pulse_pressure_mmHg']
        
        print(f"\nAortic Systolic:     {systolic:.1f} mmHg")
        print(f"  Normal range:      90 - 140 mmHg")
        if 90 <= systolic <= 140:
            print(f"  Status:            ✓ NORMAL")
            self.passes.append("Systolic pressure normal")
        else:
            print(f"  Status:            ✗ ABNORMAL")
            self.issues.append(f"Systolic pressure abnormal: {systolic:.1f} mmHg")
        
        print(f"\nAortic Diastolic:    {diastolic:.1f} mmHg")
        print(f"  Normal range:      60 - 90 mmHg")
        if 60 <= diastolic <= 90:
            print(f"  Status:            ✓ NORMAL")
            self.passes.append("Diastolic pressure normal")
        elif 50 <= diastolic < 60:
            print(f"  Status:            ⚠ SLIGHTLY LOW")
            self.warnings.append(f"Diastolic slightly low: {diastolic:.1f} mmHg")
        else:
            print(f"  Status:            ✗ ABNORMAL")
            self.issues.append(f"Diastolic pressure abnormal: {diastolic:.1f} mmHg")
        
        print(f"\nMean Arterial:       {mean_p:.1f} mmHg")
        print(f"  Normal range:      70 - 105 mmHg")
        if 70 <= mean_p <= 105:
            print(f"  Status:            ✓ NORMAL")
            self.passes.append("Mean pressure normal")
        else:
            print(f"  Status:            ✗ ABNORMAL")
            self.issues.append(f"Mean pressure abnormal: {mean_p:.1f} mmHg")
        
        print(f"\nPulse Pressure:      {pulse_p:.1f} mmHg")
        print(f"  Normal range:      40 - 60 mmHg")
        if 40 <= pulse_p <= 60:
            print(f"  Status:            ✓ NORMAL")
            self.passes.append("Pulse pressure normal")
        else:
            print(f"  Status:            ⚠ BORDERLINE")
            self.warnings.append(f"Pulse pressure borderline: {pulse_p:.1f} mmHg")
    
    def assess_cerebral_circulation(self):
        """Assess Circle of Willis perfusion"""
        print("\n" + "="*70)
        print("2. CEREBRAL CIRCULATION (Circle of Willis)")
        print("="*70)
        
        cow_balance = self.validation_data['validation_results']['cow_balance']
        inflow = cow_balance['total_inflow_ml_min']
        outflow = cow_balance['total_outflow_ml_min']
        imbalance = cow_balance['imbalance_percent']
        
        # Expected CoW flow is ~15% of cardiac output
        co = self.validation_data['validation_results']['cardiac_output']['value_l_min']
        expected_cow = co * 1000 * 0.15  # mL/min
        
        print(f"\nCoW Total Inflow:    {inflow:.2f} mL/min")
        print(f"CoW Total Outflow:   {outflow:.2f} mL/min")
        print(f"Expected CoW flow:   ~{expected_cow:.0f} mL/min (15% of CO)")
        
        if inflow < 10:
            print(f"  Status:            ✗ SEVERELY UNDERESTIMATED")
            print(f"  Problem:           Flow is ~{expected_cow/inflow:.0f}× too small!")
            self.issues.append(f"CoW flow critically low: {inflow:.2f} vs {expected_cow:.0f} mL/min expected")
        elif inflow < expected_cow * 0.5:
            print(f"  Status:            ⚠ UNDERESTIMATED")
            self.warnings.append(f"CoW flow low: {inflow:.2f} mL/min")
        else:
            print(f"  Status:            ✓ REASONABLE")
            self.passes.append("CoW flow adequate")
        
        # ICA flows
        inflow_vessels = cow_balance['inflow_vessels']
        rica = inflow_vessels.get('R-ICA', 0)
        lica = inflow_vessels.get('L-ICA', 0)
        expected_ica = expected_cow * 0.4  # Each ICA ~40% of CoW
        
        print(f"\nR-ICA flow:          {rica:.3f} mL/min")
        print(f"L-ICA flow:          {lica:.3f} mL/min")
        print(f"Expected per ICA:    ~{expected_ica:.0f} mL/min")
        
        if max(rica, lica) < 10:
            print(f"  Status:            ✗ CRITICALLY LOW")
            print(f"  Problem:           Inadequate brain perfusion")
            self.issues.append(f"ICA flows critically low: {rica:.3f}, {lica:.3f} mL/min")
        elif max(rica, lica) < expected_ica * 0.5:
            print(f"  Status:            ⚠ LOW")
            self.warnings.append(f"ICA flows low: {rica:.3f}, {lica:.3f} mL/min")
        else:
            print(f"  Status:            ✓ ADEQUATE")
            self.passes.append("ICA flows adequate")
        
        # Mass balance
        print(f"\nCoW Mass Imbalance:  {imbalance:.1f}%")
        if imbalance < 10:
            print(f"  Status:            ✓ EXCELLENT")
            self.passes.append("CoW mass balance excellent")
        elif imbalance < 25:
            print(f"  Status:            ✓ ACCEPTABLE")
            self.passes.append("CoW mass balance acceptable")
        else:
            print(f"  Status:            ⚠ LARGE")
            self.warnings.append(f"CoW imbalance large: {imbalance:.1f}%")
    
    def assess_convergence(self):
        """Assess numerical convergence"""
        print("\n" + "="*70)
        print("3. NUMERICAL CONVERGENCE")
        print("="*70)
        
        waveform = self.analysis_data['waveform_characteristics']
        periodicity = waveform['periodicity_rms_percent']
        
        print(f"\nPeriodicity RMS:     {periodicity:.2f}%")
        print(f"  Target:            < 1.0%")
        
        if periodicity < 1.0:
            print(f"  Status:            ✓ CONVERGED")
            self.passes.append("Simulation converged")
        elif periodicity < 5.0:
            print(f"  Status:            ⚠ ACCEPTABLE")
            self.warnings.append(f"Periodicity acceptable but not ideal: {periodicity:.2f}%")
        else:
            print(f"  Status:            ✗ POOR CONVERGENCE")
            print(f"  Problem:           Simulation may be in transient state")
            self.issues.append(f"Poor convergence: {periodicity:.2f}% RMS")
    
    def print_root_cause_analysis(self):
        """Print root cause analysis for issues"""
        if not self.issues:
            return
        
        print("\n" + "="*70)
        print("4. ROOT CAUSE ANALYSIS")
        print("="*70)
        
        # Check for CoW flow issues
        cow_issue = any("CoW flow" in issue or "ICA flow" in issue for issue in self.issues)
        if cow_issue:
            print("\nCerebral circulation issues detected:")
            print("  Possible causes:")
            print("    1. Abel_ref2 template has incorrect peripheral resistances")
            print("    2. Flow bypasses CoW through intermediate branches")
            print("    3. Ophthalmic, superior cerebellar branches drain early")
            print("    4. Terminal MCA/ACA/PCA outlets have excessive resistance")
            print("\n  Recommended actions:")
            print("    - Check p29-p40.csv resistance values (brain outlets)")
            print("    - Verify vessel diameters in arterial.csv for A70-A78")
            print("    - Consider using working reference (patient025_CoW_v2)")
        
        # Check for convergence issues
        conv_issue = any("convergence" in issue.lower() for issue in self.issues)
        if conv_issue:
            print("\nConvergence issues detected:")
            print("  Possible causes:")
            print("    1. Simulation time too short (< 10 cardiac cycles)")
            print("    2. Time step too large")
            print("    3. Unstable boundary conditions")
            print("\n  Recommended actions:")
            print("    - Increase simulation time in main.csv")
            print("    - Check timestep stability (CFL condition)")
            print("    - Verify heart model parameters")
    
    def print_summary(self):
        """Print assessment summary"""
        print("\n" + "="*70)
        print("BIOLOGICAL CORRECTNESS SUMMARY")
        print("="*70)
        
        global_ok = all("Cardiac output" in p or "pressure" in p for p in self.passes[:5])
        cerebral_ok = any("CoW flow" in p or "ICA" in p for p in self.passes)
        convergence_ok = any("converged" in p.lower() for p in self.passes)
        
        print(f"\n✓ PASSES:   {len(self.passes)}")
        print(f"⚠ WARNINGS: {len(self.warnings)}")
        print(f"✗ ISSUES:   {len(self.issues)}")
        
        print("\n" + "-"*70)
        
        if global_ok:
            print("✓ GLOBAL CIRCULATION: BIOLOGICALLY CORRECT")
            print("  - Cardiac output within normal range")
            print("  - Aortic pressures physiological")
            print("  - Systemic hemodynamics working properly")
        else:
            print("✗ GLOBAL CIRCULATION: ISSUES DETECTED")
            for issue in [i for i in self.issues if "Cardiac" in i or "pressure" in i]:
                print(f"  - {issue}")
        
        print()
        
        if cerebral_ok:
            print("✓ CEREBRAL CIRCULATION: BIOLOGICALLY REASONABLE")
            print("  - CoW perfusion adequate")
            print("  - Brain receives sufficient blood flow")
        else:
            print("✗ CEREBRAL CIRCULATION: NOT BIOLOGICALLY REALISTIC")
            print("  - CoW flow severely underestimated")
            print("  - Brain perfusion inadequate")
            print("  - This is a known Abel_ref2 template limitation")
        
        print()
        
        if convergence_ok:
            print("✓ CONVERGENCE: ACHIEVED")
            print("  - Simulation reached steady periodic state")
        else:
            print("⚠ CONVERGENCE: QUESTIONABLE")
            if self.warnings and any("Periodicity" in w for w in self.warnings):
                print("  - Acceptable but not ideal convergence")
            else:
                print("  - May need longer simulation time")
        
        print("\n" + "-"*70)
        print("OVERALL VERDICT:")
        print("-"*70)
        
        if len(self.issues) == 0 and len(self.warnings) <= 2:
            print("✓ SIMULATION IS BIOLOGICALLY VALID")
            print("  Suitable for publication and clinical interpretation")
        elif global_ok and not cerebral_ok:
            print("⚠ PARTIALLY VALID")
            print("  ✓ Valid for: Global hemodynamic analysis")
            print("  ✗ NOT valid for: Cerebral perfusion studies")
            print("  Note: This is expected with Abel_ref2 template")
        else:
            print("✗ SIMULATION HAS SIGNIFICANT ISSUES")
            print("  Review root cause analysis and fix before proceeding")
        
        print("="*70)
    
    def run_assessment(self):
        """Run complete biological assessment"""
        print("="*70)
        print(f"BIOLOGICAL ASSESSMENT: {self.model_name}")
        print("="*70)
        
        if not self.load_data():
            return False
        
        self.assess_global_hemodynamics()
        self.assess_cerebral_circulation()
        self.assess_convergence()
        self.print_root_cause_analysis()
        self.print_summary()
        
        return len(self.issues) == 0


def main():
    """Run biological assessment"""
    parser = argparse.ArgumentParser(
        description="Assess biological correctness of FirstBlood simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 biological_assessment.py                    # Interactive prompt
    python3 biological_assessment.py --model patient_025
    python3 biological_assessment.py --models patient_025,Abel_ref2
    python3 biological_assessment.py --all

Requirements:
    Must run validation.py and analysis.py first for each model.
        """
    )
    parser.add_argument('--model',
                       help='Single model name (e.g., patient_025, Abel_ref2)')
    parser.add_argument('--models',
                       help='Comma-separated model names')
    parser.add_argument('--models-file',
                       help='Path to text file with one model name per line')
    parser.add_argument('--all', action='store_true',
                       help='Assess all models with validation data')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory (default: pipeline/output)')
    
    args = parser.parse_args()
    
    # Set up paths
    repo_root = get_repo_root()
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "pipeline" / "output"
    
    # Resolve model list
    model_names = []
    if args.model:
        model_names.append(args.model)
    if args.models:
        model_names.extend([m.strip() for m in args.models.split(',') if m.strip()])
    if args.models_file:
        models_file = Path(args.models_file)
        if not models_file.exists():
            print(f"[ERROR] models-file not found: {models_file}")
            sys.exit(1)
        with open(models_file, 'r') as f:
            model_names.extend([line.strip() for line in f if line.strip()])
    if args.all:
        validation_dir = output_dir / "validation"
        if validation_dir.exists():
            model_names.extend([
                f.stem.replace('_validation', '') 
                for f in validation_dir.glob('*_validation.json')
            ])
    
    # De-duplicate
    seen = set()
    model_names = [m for m in model_names if not (m in seen or seen.add(m))]
    
    if not model_names:
        try:
            entered = input("Enter model name for assessment (e.g., patient_025): ").strip()
        except EOFError:
            entered = ""
        if entered:
            model_names.append(entered)
        else:
            print("[ERROR] No models specified.")
            sys.exit(1)
    
    overall_success = True
    
    for model_name in model_names:
        assessor = BiologicalAssessor(model_name, output_dir)
        success = assessor.run_assessment()
        if not success:
            overall_success = False
        print("\n")
    
    return 0 if overall_success else 1


if __name__ == '__main__':
    sys.exit(main())
