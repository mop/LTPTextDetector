import numpy as np
import subprocess
import re
import os

TEMPLATE = """%%YAML:1.0
input_directory: ../sample_icdar_2005/
responses_directory: ../result_sample/
crf_model_file: models/model_crf_rf_svm.dat
svm_model_file: models/model_svm_cc.yml
cnn_model_file: dumpnet_bin/
random_forest_model_file: models/model_forest_cc.yml
random_forest_pw_1_1_model_file: models/model_forest_cc_pw_1_1.yml
random_forest_pw_1_0_model_file: models/model_forest_cc_pw_1_0.yml
random_forest_pw_0_0_model_file: models/model_forest_cc_pw_0_0.yml
random_seed: 4
threshold: 0.10
word_split_model: "MODEL_PROJECTION_PROFILE_SOFT"
classification_model: "CLASSIFICATION_MODEL_CRF_RF"
pre_classification_model: "PRE_CLASSIFICATION_MODEL_RF_SVM_ENSEMBLE"
word_group_threshold: %(word_group_threshold)f
pre_classification_prob_threshold: %(pre_classification_prob_threshold)f
maximum_height_ratio: %(maximum_height_ratio)f
minimum_vertical_overlap: %(minimum_vertical_overlap)f
allow_single_letters: 1
cache_directory: "cache_rf"
verbose: 1
"""

os.system("taskset -p 0xff %d >/dev/null" % os.getpid())
LOGFILE = 'gridlog.txt'
log = open(LOGFILE, 'w')

word_group_grid = [x for x in np.linspace(0.5, 1.0, 6)]
pre_class_group_grid = [x for x in np.linspace(0, 0.5, 6)]
min_vertical_overlap_grid = [x for x in np.linspace(0.2, 0.8, 7)]
max_height_ratio_grid = [x for x in np.linspace(1, 3, 9)]
params = [(w,p,mver,maxh) for w in word_group_grid for p in pre_class_group_grid for mver in min_vertical_overlap_grid for maxh in max_height_ratio_grid]
np.random.shuffle(params)

best_params = []
best_fscore = 0
for w,p,mver,maxh in params:
    template = TEMPLATE % { 
        'word_group_threshold': w,
        'pre_classification_prob_threshold': p,
        'maximum_height_ratio': maxh,
        'minimum_vertical_overlap': mver
    }
    with open('config_cv.yml', 'w') as fp:
        fp.write(template)
    os.system('./bin/create_boxes -c config_cv.yml')
    os.system('cd .. && python2 ./scripts/to_xml4.py result_sample > eval_cv.xml')
    os.system('cd .. && /home/nax/Downloads/deteval-linux/detevalcmd/evaldetection eval_cv.xml datasets/icdar-sample/locations.xml > results_cv.xml')
    pipe = subprocess.Popen('cd .. && /home/nax/Downloads/deteval-linux/detevalcmd/readdeteval results_cv.xml', shell=True, stdout=subprocess.PIPE)
    result_str = pipe.stdout.read()
    hmean_2003, hmean_2011 = re.findall('hmean="([^"]*)"', result_str, flags=re.MULTILINE)
    hmean_2003, hmean_2011 = float(hmean_2003), float(hmean_2011)
    if hmean_2003 > best_fscore:
        best_fscore = hmean_2003
        best_params = (w,p, mver, maxh)
        print '****NEW BEST****'
        print best_fscore
        print best_params
        os.system('cp ../eval_cv.xml ../eval_cv_best.xml')
    log.write("F-2003: %f, F-2011: %f, w: %f, p: %f, minv: %f, maxh: %f\n" % (hmean_2003, hmean_2011, w,p, mver, maxh))
    log.flush()

print '****BEST****'
print best_fscore
print best_params
log.close()
