source ./scripts/config.sh
mkdir -p $DATA_DIR/{orig,tmp}
mkdir $MODEL_DIR

./bin/extract_hog_features -f -i ../train_icdar_2005/ -g ../train_icdar_2005_mser_cc -o $DATA_DIR/orig/features_hog.csv -b $DATA_DIR/orig/features_bin.csv
./bin/extract_hog_features -f -i ../test_icdar_2005/ -g ../test_icdar_2005_mser_cc -o $DATA_DIR/orig/features_hog_test.csv -b $DATA_DIR/orig/features_bin_test.csv
