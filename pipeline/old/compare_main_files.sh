#!/bin/bash
echo "Comparing main.csv files in detail:"
echo "====================================="
echo ""

echo "Abel_ref2 main.csv:"
cat ~/first_blood/models/Abel_ref2/main.csv

echo ""
echo "=========================================="
echo ""

echo "patient_025 main.csv:"
cat ~/first_blood/models/patient_025/main.csv

echo ""
echo "=========================================="
echo "Diff between them:"
echo "=========================================="
diff ~/first_blood/models/Abel_ref2/main.csv ~/first_blood/models/patient_025/main.csv