#!/bin/bash
echo "Comparing patient_025 with Abel_ref2:"
echo "======================================"
echo ""

echo "Files in Abel_ref2:"
ls ~/first_blood/models/Abel_ref2/*.csv | wc -l
echo "  CSV files"

echo ""
echo "Files in patient_025:"
ls ~/first_blood/models/patient_025/*.csv | wc -l
echo "  CSV files"

echo ""
echo "Critical files comparison:"
echo "-------------------------"
for file in arterial.csv main.csv heart_kim_lit.csv; do
    printf "%-20s" "$file:"
    if [ -f ~/first_blood/models/Abel_ref2/$file ]; then
        printf " Abel_ref2: YES"
    else
        printf " Abel_ref2: NO "
    fi
    if [ -f ~/first_blood/models/patient_025/$file ]; then
        printf "  patient_025: YES"
    else
        printf "  patient_025: NO "
    fi
    echo ""
done

echo ""
echo "Checking main.csv format:"
echo "------------------------"
echo "Abel_ref2 main.csv first 5 lines:"
head -5 ~/first_blood/models/Abel_ref2/main.csv

echo ""
echo "patient_025 main.csv first 5 lines:"
head -5 ~/first_blood/models/patient_025/main.csv

echo ""
echo "Checking if simulation can find files:"
cd ~/first_blood/projects/simple_run
echo "Current directory: $(pwd)"
echo "Relative path to patient_025: ../../models/patient_025"
echo "Does arterial.csv exist at that path?"
ls -l ../../models/patient_025/arterial.csv 2>&1 | head -1
echo ""
echo "Does main.csv exist at that path?"
ls -l ../../models/patient_025/main.csv 2>&1 | head -1

echo ""
echo "Testing Abel_ref2 (should work):"
./simple_run.out Abel_ref2 2>&1 | head -5