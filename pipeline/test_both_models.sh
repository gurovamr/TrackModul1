#!/bin/bash
echo "=========================================="
echo "Testing Abel_ref2 (should work):"
echo "=========================================="
cd ~/first_blood/projects/simple_run
./simple_run.out Abel_ref2 2>&1 | head -30

echo ""
echo "=========================================="
echo "Testing patient_025:"
echo "=========================================="
./simple_run.out patient_025 2>&1 | head -30

echo ""
echo "=========================================="
echo "Comparing file lists:"
echo "=========================================="
echo "Files ONLY in Abel_ref2:"
comm -23 <(ls ~/first_blood/models/Abel_ref2/*.csv | xargs -n1 basename | sort) \
         <(ls ~/first_blood/models/patient_025/*.csv | xargs -n1 basename | sort)

echo ""
echo "Files ONLY in patient_025:"
comm -13 <(ls ~/first_blood/models/Abel_ref2/*.csv | xargs -n1 basename | sort) \
         <(ls ~/first_blood/models/patient_025/*.csv | xargs -n1 basename | sort)