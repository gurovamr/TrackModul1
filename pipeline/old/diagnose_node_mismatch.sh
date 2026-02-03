#!/bin/bash
echo "=========================================="
echo "DIAGNOSING NODE MISMATCH"
echo "=========================================="
echo ""

echo "1. Nodes declared in arterial.csv:"
echo "-----------------------------------"
grep "^node," ~/first_blood/models/patient_025/arterial.csv | head -20

echo ""
echo "2. Terminal nodes (end_node) in arterial.csv:"
echo "----------------------------------------------"
awk -F, 'NR>1 {print $5}' ~/first_blood/models/patient_025/arterial.csv | sort -u | grep "^p" | head -20

echo ""
echo "3. Nodes declared in main.csv:"
echo "------------------------------"
grep "^node," ~/first_blood/models/patient_025/main.csv | head -10

echo ""
echo "4. Peripheral mappings in main.csv:"
echo "-----------------------------------"
grep "^lumped,p" ~/first_blood/models/patient_025/main.csv | head -10

echo ""
echo "5. Comparison with Abel_ref2:"
echo "-----------------------------"
echo "Abel_ref2 arterial.csv node declarations:"
grep "^node," ~/first_blood/models/Abel_ref2/arterial.csv | head -5

echo ""
echo "Abel_ref2 arterial.csv peripheral terminals:"
awk -F, 'NR>1 {print $5}' ~/first_blood/models/Abel_ref2/arterial.csv | sort -u | grep "^p" | head -5

echo ""
echo "=========================================="
echo "DIAGNOSIS:"
echo "=========================================="
echo "The problem: arterial.csv does NOT declare nodes."
echo "In first_blood, nodes are implicitly defined by vessel endpoints."
echo ""
echo "The arterial.csv should NOT have 'node,' lines at all!"
echo "Only main.csv should declare nodes."