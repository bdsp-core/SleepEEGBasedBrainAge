export PYTHON=python3
export FLAG="-W ignore"
#$PYTHON $FLAG step1_generate_mastersheet.py
$PYTHON $FLAG step2_generate_features.py && $PYTHON $FLAG step3_generate_spindle_features.py && $PYTHON $FLAG step5_compute_stable_BA.py
#&& $PYTHON $FLAG step4_compute_BA.py
find -type l -delete

