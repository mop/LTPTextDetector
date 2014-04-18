source scripts/config.sh
./bin/extract_train_set -i ../train_icdar_2005 -g ../train_boxes/ -n 350 -o $DATA_DIR/boost/train.txt
python2 ../scripts/subsample.py $DATA_DIR/boost/train.txt '^1' 15000 > $DATA_DIR/boost/train_pos_rnd.txt
python2 ../scripts/subsample.py $DATA_DIR/boost/train.txt '^-1' 15000 > $DATA_DIR/boost/train_neg_rnd.txt
head -n 5000 $DATA_DIR/boost/train_pos_rnd.txt > $DATA_DIR/boost/train_working.txt
head -n 5000 $DATA_DIR/boost/train_neg_rnd.txt >> $DATA_DIR/boost/train_working.txt
tail -n 5000 $DATA_DIR/boost/train_pos_rnd.txt > $DATA_DIR/boost/valid_working.txt
tail -n 5000 $DATA_DIR/boost/train_neg_rnd.txt >> $DATA_DIR/boost/valid_working.txt
#python2 ./conv.py
$BOOST_BIN_DIR/rf -c models/config_boost.ini
./bin/classify -t ../train_extra -r ../result_extra -m $MODEL_DIR/model_boost.txt -u 350 -f $DATA_DIR/boost/false_positives.txt
python2 ../scripts/subsample.py $DATA_DIR/boost/false_positives.txt '.*' 10000 > $DATA_DIR/boost/fps.txt
#python2 conv2.py
cat $DATA_DIR/boost/fps.txt | cut -d ',' -f 6- | sed 's/.*/-1,&/' > $DATA_DIR/boost/fps_fin.txt
cat $DATA_DIR/boost/fps_fin.txt >> $DATA_DIR/boost/train_working.txt
$BOOST_BIN_DIR/rf -c models/config_boost.ini
./bin/classify -t ../valid_icdar_2005 -r ../result_valid -m $MODEL_DIR/model_boost.txt 
