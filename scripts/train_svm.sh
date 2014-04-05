source ./scripts/config.sh

python2 ./scripts/do_real_svm.py --w1 $w1 --gamma $g --c $C --input $DATA_DIR/orig/data_cc_ellipse.csv --output-norm $DATA_DIR/tmp/data_ellipse_cc_scaled.csv --output-cv-results $DATA_DIR/tmp/results_cv_svm.csv --output-libsvm $DATA_DIR/tmp/data_ellipse_scaled.dat --output-means $MODEL_DIR/data_cc_ellipse_means.csv --output-stds $MODEL_DIR/data_cc_ellipse_stds.csv
python2 ./scripts/fit_sgd_cv.py --input $DATA_DIR/orig/data_cc_ellipse.csv --input-svm-cv-results $DATA_DIR/tmp/results_cv_svm.csv --output-cv-results $DATA_DIR/tmp/results_cv_svm_probs.csv --output-params $MODEL_DIR/svm_sigmoid_scale.csv 

$PATH_TO_LIBSVM/svm-train -w1 $w1 -g $g -c $C $DATA_DIR/tmp/data_ellipse_scaled.dat $MODEL_DIR/svm_cc_model.dat
