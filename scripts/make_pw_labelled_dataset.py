import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Make a pairwise labelled dataset for random forests')
parser.add_argument('--unary', dest='unary_filename', help='the path to the unary feature file as extracted form extract_cc_features')
parser.add_argument('--pairwise', dest='pairwise_filename', help='the path to the pairwise feature file as extracted form extract_cc_features')
parser.add_argument('--labels', dest='labels', help='either 1, 0 or neq')
parser.add_argument('--output', dest='output_filename', help='the output filename')

args = parser.parse_args()
if not args.unary_filename or not args.pairwise_filename or not args.labels or not args.output_filename:
    parser.print_help()
    sys.exit(1)

label = args.labels
if label not in ['1', '0', 'neq']:
    parser.print_help()
    sys.exit(1)

unary    = np.genfromtxt(args.unary_filename, delimiter=',')
pairwise = np.genfromtxt(args.pairwise_filename, delimiter=',')

# build the label map
label_map = {}
for i in xrange(unary.shape[0]):
    key = '%s-%s' % (int(unary[i,0]), int(unary[i,1]))
    label_map[key] = unary[i,2]

result = np.zeros((pairwise.shape[0], pairwise.shape[1] - 3 + 1))
result[:,1:] = pairwise[:,3:]
for i in xrange(pairwise.shape[0]):
    imgid = int(pairwise[i,0])
    uid1 = int(pairwise[i,1])
    uid2 = int(pairwise[i,2])

    lbl_1 = int(label_map['%s-%s' % (imgid, uid1)])
    lbl_2 = int(label_map['%s-%s' % (imgid, uid2)])

    the_label = 1
    if label == 'neq':
        the_label = -1 if lbl_1 != lbl_2 else 1
    elif label == '1':
        the_label = 1 if lbl_1 > 0 and lbl_2 > 0 else -1
    elif label == '0':
        the_label = -1 if lbl_1 <= 0 and lbl_2 <= 0 else 1
        
    result[i,0] = the_label
np.savetxt(args.output_filename, result, delimiter=',', fmt='%f')
