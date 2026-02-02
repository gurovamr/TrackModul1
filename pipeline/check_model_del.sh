#!/bin/bash
echo "Checking patient_025 model directory:"
echo "======================================"
echo ""

if [ ! -d ~/first_blood/models/patient_025 ]; then
    echo "ERROR: Directory does not exist!"
    exit 1
fi

echo "Files in ~/first_blood/models/patient_025/:"
ls -lh ~/first_blood/models/patient_025/ | head -20
echo ""
echo "Total CSV files:"
ls ~/first_blood/models/patient_025/*.csv 2>/dev/null | wc -l
echo ""

echo "Checking for critical files:"
for file in parameter.csv arterial.csv main.csv heart_kim_lit.csv p1.csv p47.csv; do
    if [ -f ~/first_blood/models/patient_025/$file ]; then
        size=$(stat -f%z ~/first_blood/models/patient_025/$file 2>/dev/null || stat -c%s ~/first_blood/models/patient_025/$file 2>/dev/null)
        echo "  [OK] $file ($size bytes)"
    else
        echo "  [MISSING] $file"
    fi
done
echo ""

echo "Checking Abel_ref2 template:"
if [ -d ~/first_blood/models/Abel_ref2 ]; then
    echo "  [OK] Abel_ref2 exists"
    echo "  Files in template:"
    ls ~/first_blood/models/Abel_ref2/*.csv 2>/dev/null | wc -l
    echo "  CSV files"
else
    echo "  [ERROR] Abel_ref2 NOT FOUND"
fi