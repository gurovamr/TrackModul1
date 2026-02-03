#!/bin/bash
echo "=========================================="
echo "ANATOMICAL VALIDATION: Patient 025"
echo "=========================================="
echo ""

echo "1. CoW vessels modified (from modifications_log.csv):"
echo "------------------------------------------------------"
if [ -f ~/first_blood/models/patient_025/modifications_log.csv ]; then
    head -20 ~/first_blood/models/patient_025/modifications_log.csv | column -t -s,
else
    echo "  modifications_log.csv not found"
fi

echo ""
echo "=========================================="
echo "2. CoW-specific vessels in arterial.csv:"
echo "=========================================="
cd ~/first_blood/models/patient_025
grep -E "A60|A61|A62|A63|A64|A65|A68|A69|A76|A77|A78|Pcom|cerebral|communicating" arterial.csv | head -20

echo ""
echo "=========================================="
echo "3. Patient 025 topology from raw data:"
echo "=========================================="
echo "From topcow_ct_025.json variant file:"
cat ~/first_blood/data/cow_variants/topcow_ct_025.json 2>/dev/null || echo "Variant file not found"

echo ""
echo "=========================================="
echo "4. Critical question: Peripheral mapping"
echo "=========================================="
echo "The arterial.csv has these CoW terminal nodes:"
grep -E "cerebral 2|Post\. cerebral 2|Ant\. cerebral A2" ~/first_blood/models/patient_025/arterial.csv | \
  awk -F, '{print $2 " (ID: " $1 "): " $4 " -> " $5}'

echo ""
echo "These should be:"
echo "  - R-PCA terminal (A64): n47 -> p36"
echo "  - L-PCA terminal (A65): n48 -> p33"  
echo "  - R-ACA terminal (A76): n42 -> p38"
echo "  - L-ACA terminal (A78): n41 -> p31"
echo ""
echo "Do these match the patient's anatomy?"