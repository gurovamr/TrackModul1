#!/bin/bash
# Remove the erroneous "parameter" entries from main.csv

echo "Fixing patient_025 main.csv..."
echo ""

cd ~/first_blood/models/patient_025

# Backup original
cp main.csv main.csv.backup

# Remove the parameter entries
grep -v "parameter" main.csv.backup > main.csv

echo "Fixed! Removed these lines:"
diff main.csv.backup main.csv | grep "parameter"

echo ""
echo "Verification - checking for 'parameter' in new main.csv:"
if grep -q "parameter" main.csv; then
    echo "  ERROR: Still found 'parameter' in main.csv"
else
    echo "  OK: No 'parameter' entries found"
fi

echo ""
echo "Now try running:"
echo "  cd ~/first_blood/projects/simple_run"
echo "  ./simple_run.out patient_025"