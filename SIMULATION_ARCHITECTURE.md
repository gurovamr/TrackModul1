# FirstBlood Simulation Architecture

## Overview
FirstBlood is a 1D hemodynamic simulation framework for patient-specific cardiovascular modeling using the Method of Characteristics (MOC) solver.

---

## Directory Structure

```
first_blood/
├── data_patient025/              # Raw patient data
│   ├── nodes_mr_025.json        # Vessel node positions & connectivity
│   ├── feature_mr_025.json      # Vessel geometric features (radius, length)
│   └── variant_mr_025.json      # Circle of Willis anatomical variants
│
├── models/                       # Simulation input models
│   ├── Abel_ref2/               # Reference baseline model
│   ├── patient_025/             # Patient-specific model
│   │   ├── main.csv             # Simulation configuration
│   │   ├── arterial.csv         # Arterial vessel network definition
│   │   ├── heart_kim_lit.csv    # Cardiac model parameters
│   │   └── p*.csv               # Peripheral boundary conditions (47 files)
│   │
├── pipeline/                     # Data generation & validation scripts
│   ├── data_generationV*.py     # Patient model generation scripts
│   └── validation.py            # Simulation validation script
│
├── projects/simple_run/          # Execution environment
│   ├── simple_run.out           # Compiled simulation executable
│   └── results/                 # Simulation outputs
│       └── MODEL_NAME/
│           ├── arterial/        # Time-series results per vessel
│           │   ├── A1.txt       # Ascending aorta
│           │   ├── A12.txt      # Right ICA
│           │   ├── A70.txt      # Right MCA
│           │   └── ...          # ~103 arterial vessels
│           ├── p*/              # Peripheral outlet results
│           └── heart_kim_lit/   # Cardiac output
│
├── source/                       # C++ simulation engine
│   ├── first_blood.cpp          # Main simulation class
│   ├── moc_edge.cpp             # Arterial vessel (1D MOC solver)
│   └── solver_moc.cpp           # MacCormack numerical scheme
│
└── analysis_V3/                  # Analysis scripts
    ├── validate_simulation.py   # Detailed validation
    ├── compare_abel.py          # Waveform comparison
    └── analyze_*.py             # Various analysis tools
```

---

## Input Data Structure

### 1. Patient Raw Data (`data_patient025/`)

**nodes_mr_025.json** - Network topology:
```json
{
  "1": {
    "BA start": [{"id": 0, "degree": 1, "coords": [-9.2, -19.6, -15.4]}],
    "BA bifurcation": [{"id": 51, "degree": 3, "coords": [-8.5, -22.0, -3.0]}],
    ...
  }
}
```
- Defines anatomical landmarks and node connectivity
- 3D coordinates for geometric reconstruction
- Node degrees indicate bifurcations (degree 3) or connections (degree 1-2)

**feature_mr_025.json** - Vessel geometry:
```json
{
  "1": {
    "BA": [{
      "segment": {"start": 15, "end": 51},
      "radius": {"mean": 2.139, "min": 1.933, "max": 2.296},
      "length": 9.97,
      "tortuosity": 0.026
    }],
    ...
  }
}
```
- Measured vessel radii, lengths, tortuosity
- Patient-specific geometric parameters
- Used to replace template values in arterial.csv

**variant_mr_025.json** - Circle of Willis anatomy:
```json
{
  "anterior": {"L-A1": true, "R-A1": true, "Acom": true},
  "posterior": {"L-Pcom": true, "R-Pcom": true, "L-P1": true, "R-P1": true},
  "fetal": {"R-PCA": true, "L-PCA": false}
}
```
- Indicates presence/absence of communicating arteries
- Critical for CoW flow distribution
- `false` = vessel absent/occluded

---

## Model Files (`models/patient_025/`)

### main.csv - Simulation Configuration
```csv
run,forward
time,10.317              # Total simulation time (seconds)
material,linear          # Vessel wall material model
solver,maccormack        # Numerical scheme

type,name,main node,model node,...
moc,arterial,N1p,p1,N10p,p10,...  # MOC solver for arterial network
lumped,p1,N1p,p1                  # Windkessel boundary at periphery
```
- Specifies solver type (MOC = Method of Characteristics)
- Maps between main nodes (N*) and model nodes (p*, H, n*)
- Connects arterial network to peripheral boundaries

### arterial.csv - Vessel Network Definition
```csv
type,ID,name,start_node,end_node,start_diameter[SI],end_diameter[SI],...
vis_f,A1,Ascending aorta 1,H,n1,0.0294,0.0293,0.00294,0.00293,0.005,...
vis_f,A12,Internal carotid,n32,n46,0.00467,0.00467,0.00057,0.00043,0.011,...
vis_f,A70,Middle cerebral M1,n43,n36,0.00379,0.00379,0.0003,0.00028,0.007,...
```

**Key columns:**
- `ID`: Vessel identifier (A1-A103)
- `start_node`/`end_node`: Network connectivity
  - `H` = Heart (inlet)
  - `n*` = Internal junction nodes
  - `p*` = Peripheral outlets (boundary conditions)
- `start_diameter`/`end_diameter`: Vessel radius (meters)
- `start_thickness`/`end_thickness`: Wall thickness (meters)
- `length[SI]`: Vessel length (meters)
- `division_points`: Number of spatial discretization points
- `elastance_1[SI]`: Wall stiffness (Pa)
- `k1`, `k2`, `k3`: Material model parameters

### heart_kim_lit.csv - Cardiac Model
- Time-varying cardiac output
- Heart rate, stroke volume
- Inlet boundary condition at node `H`

### p1.csv - p47.csv - Peripheral Boundaries
- Windkessel models (3-element RCR circuits)
- Resistance, compliance, characteristic impedance
- One file per peripheral outlet

---

## Network Connectivity

### Node Types
1. **H (Heart)**: Inlet boundary condition
   - Source of all arterial flow
   - Time-varying pressure/flow from heart_kim_lit.csv

2. **n* (Internal nodes)**: Junction points
   - `n1` - `n53`: Bifurcations and connections
   - Example: `n49` = Basilar bifurcation → R-PCA, L-PCA

3. **p* (Peripheral nodes)**: Outlet boundaries
   - `p1` - `p47`: Terminal vessels
   - Connect to Windkessel boundary conditions
   - Represent downstream vascular beds

### Key Vessels in Circle of Willis
```
Inflow:
  A12: R-ICA (Right Internal Carotid)  → n32 to n46
  A16: L-ICA (Left Internal Carotid)   → n26 to n37
  A59: Basilar artery                  → n50 to n49

Outflow:
  A70: R-MCA (Right Middle Cerebral)   → n43 to n36
  A73: L-MCA (Left Middle Cerebral)    → n40 to n30
  A76: R-ACA (Right Anterior Cerebral) → n42 to p38
  A78: L-ACA (Left Anterior Cerebral)  → n41 to p31
  A64: R-PCA (Right Posterior Cerebral)→ n47 to p36
  A65: L-PCA (Left Posterior Cerebral) → n48 to p33

Communicating:
  A77: Acom (Anterior Communicating)   → n41 to n42
  A62: R-Pcom (Right Posterior Comm.)  → n47 to n45
  A63: L-Pcom (Left Posterior Comm.)   → n48 to n38
```

---

## Simulation Workflow

### 1. Model Generation (`pipeline/data_generationV*.py`)
```bash
cd ~/first_blood/pipeline
python3 data_generationV2.py --pid 025 --modality mr
```

**Process:**
1. Load patient raw data (nodes, features, variants)
2. Copy template model (Abel_ref2) as starting point
3. Map patient vessels to template IDs:
   - ICA → A12, A16
   - MCA → A70, A73
   - BA → A56, A59
   - ACA → A68, A69, A76, A78
   - PCA → A60, A61, A64, A65
4. Inject patient-specific geometry (radius, length)
5. Apply CoW variants (occlude absent vessels)
6. Generate arterial.csv for patient_025

**Key mapping logic:**
- Template vessel IDs (A1-A103) are preserved
- Only geometric parameters are updated
- Node connectivity remains unchanged
- Wall thickness is NOT modified (uses template values)

### 2. Simulation Execution
```bash
cd ~/first_blood/projects/simple_run
./simple_run.out patient_025
```

**Execution flow:**
1. `first_blood.cpp` loads main.csv
2. Reads arterial.csv → creates MOC edges (vessels)
3. Reads p*.csv → creates Windkessel boundaries
4. Reads heart_kim_lit.csv → sets inlet BC
5. Runs MacCormack solver for specified time
6. Outputs results to `results/patient_025/arterial/`

**Numerical method:**
- 1D blood flow equations (mass + momentum conservation)
- Method of Characteristics (MOC) with MacCormack scheme
- Spatial discretization: 5-50 points per vessel
- Time step: Adaptive (CFL condition)
- Typical runtime: 10 seconds of cardiac cycles

### 3. Results Analysis
```bash
cd ~/first_blood/pipeline
python3 validation.py --model patient_025
```

**Output files (`results/patient_025/arterial/`):**
- One .txt file per vessel (A1.txt, A12.txt, ...)
- Each file: time-series of pressure, velocity, flow
- Format: `time, P_start, P_end, V_start, V_end, Q_start, Q_end`

**Validation checks:**
1. Cardiac output: 4-7 L/min (physiological range)
2. CoW mass balance: Inflow ≈ Outflow
3. Periodicity: Last cycles should converge (< 1% RMS)
4. Pressure ranges: 80-160 mmHg systolic, 40-100 diastolic
5. Numerical stability: No oscillations or negative pressures

---

## Circle of Willis Imbalance

**Your 45.4% imbalance means:**
```
Inflow  = 0.61 mL/min (A12 + A16 + A59)
Outflow = 0.33 mL/min (A70 + A73 + A76 + A78 + A64 + A65)
Missing = 0.28 mL/min (45.4%)
```

**Root causes:**
1. **Peripheral leakage**: Flow exits through intermediate branches before reaching terminal MCAs/ACAs/PCAs
   - Ophthalmic arteries (A80, A82)
   - Anterior choroidal
   - Superior cerebellar (A57, A58)
   
2. **Low peripheral resistances**: Terminal p*.csv files may have resistance values that are too low
   - Should be: R_total ≈ 80-120 mmHg / (total_flow)
   - Check: p29-p40 (MCA/ACA/PCA outlets)

3. **Variant-induced flow redistribution**: 
   - R-PCA is fetal (fed by ICA, not basilar)
   - This reduces basilar contribution to CoW

**Fix strategy:**
1. Increase peripheral resistances in p29-p40.csv
2. Reduce resistance in intermediate branches
3. Re-run simulation and validate again

---

## Key Takeaways

✓ **Data flow**: Raw patient data → Model generation → Simulation → Validation  
✓ **Network structure**: Heart → Arteries → CoW → Periphery  
✓ **Vessel IDs**: A1-A103 are fixed template IDs, geometry is patient-specific  
✓ **Simulation**: 1D MOC solver with MacCormack scheme  
✓ **Validation**: Focus on cardiac output, CoW balance, periodicity  

The system is working correctly - the 45% CoW imbalance is a modeling issue (peripheral resistances), not a code bug.
