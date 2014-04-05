#!/bin/bash

source ./scripts/config.sh
# activate the virtualenv
source $VIRTUAL_ENV_DIR

mkdir -p $DATA_DIR/{orig,tmp}
mkdir $MODEL_DIR

# extract the features
./bin/extract_cc_features -f -i ../train_icdar_2005/ -g ../train_icdar_2005_mser_cc -p $DATA_DIR/orig/features_pairwise.csv -o $DATA_DIR/orig/features_unary.csv

# prepare the datasets for RF
cut -d ',' -f 3- $DATA_DIR/orig/features_unary.csv > $DATA_DIR/orig/data_cc_ellipse.csv
python scripts/make_pw_labelled_dataset.py --unary $DATA_DIR/orig/features_unary.csv --pairwise $DATA_DIR/orig/features_pairwise.csv --labels 1 --output $DATA_DIR/orig/features_pairwise_1_1.csv
python scripts/make_pw_labelled_dataset.py --unary $DATA_DIR/orig/features_unary.csv --pairwise $DATA_DIR/orig/features_pairwise.csv --labels neq --output $DATA_DIR/orig/features_pairwise_1_0.csv
python scripts/make_pw_labelled_dataset.py --unary $DATA_DIR/orig/features_unary.csv --pairwise $DATA_DIR/orig/features_pairwise.csv --labels 0 --output $DATA_DIR/orig/features_pairwise_0_0.csv

# train the forests
./bin/train_forest -i $DATA_DIR/orig/data_cc_ellipse.csv -o $MODEL_DIR/model_forest_cc.yml -n 100 -d 40 -v 5 -c 1 
./bin/train_forest -i $DATA_DIR/orig/features_pairwise_1_0.csv -o $MODEL_DIR/model_forest_cc_pw_1_0.yml -n 100 -d 40 -v 3
./bin/train_forest -i $DATA_DIR/orig/features_pairwise_1_1.csv -o $MODEL_DIR/model_forest_cc_pw_1_1.yml -n 100 -d 40 -v 3
./bin/train_forest -i $DATA_DIR/orig/features_pairwise_0_0.csv -o $MODEL_DIR/model_forest_cc_pw_0_0.yml -n 100 -d 40 -v 3

# create the predictions
./bin/cv_predict_forest -i $DATA_DIR/orig/features_pairwise_1_1.csv -d '25' -f 10 -n '100' -v '3' -o $DATA_DIR/tmp/cv_results_pw_1_1.csv
./bin/cv_predict_forest -i $DATA_DIR/orig/features_pairwise_1_0.csv -d '25' -f 10 -n '100' -v '3' -o $DATA_DIR/tmp/cv_results_pw_1_0.csv
./bin/cv_predict_forest -i $DATA_DIR/orig/features_pairwise_0_0.csv -d '25' -f 10 -n '100' -v '3' -o $DATA_DIR/tmp/cv_results_pw_0_0.csv
./bin/cv_predict_forest -i $DATA_DIR/orig/data_cc_ellipse.csv -d '40' -f 10 -n '100' -v '5' -o $DATA_DIR/tmp/cv_results_un.csv

./scripts/train_svm.sh

# prepare the CRF predictions
#python scripts/prepare_crf.py --unary $DATA_DIR/orig/features_unary.csv --pairwise $DATA_DIR/orig/features_pairwise.csv --unary-results $DATA_DIR/tmp/cv_results_un.csv,$DATA_DIR/tmp/results_cv_svm_probs.csv --pairwise-results $DATA_DIR/tmp/cv_results_pw_1_1.csv,$DATA_DIR/tmp/cv_results_pw_1_0.csv,$DATA_DIR/tmp/cv_results_pw_0_0.csv --unary-output $DATA_DIR/tmp/features_unary_rf.csv --pairwise-output $DATA_DIR/tmp/features_pairwise_rf.csv
python scripts/prepare_crf.py --unary $DATA_DIR/orig/features_unary.csv --pairwise $DATA_DIR/orig/features_pairwise.csv --unary-results $DATA_DIR/tmp/cv_results_un.csv --pairwise-results $DATA_DIR/tmp/cv_results_pw_1_1.csv,$DATA_DIR/tmp/cv_results_pw_1_0.csv,$DATA_DIR/tmp/cv_results_pw_0_0.csv --unary-output $DATA_DIR/tmp/features_unary_rf.csv --pairwise-output $DATA_DIR/tmp/features_pairwise_rf.csv

# train the crf
./bin/train_crf2 -i $DATA_DIR/tmp/features_unary_rf.csv -p $DATA_DIR/tmp/features_pairwise_rf.csv -o $MODEL_DIR/model_crf_rf.dat
