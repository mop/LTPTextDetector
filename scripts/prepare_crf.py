import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Make a pairwise labelled dataset for random forests')
parser.add_argument('--unary', dest='unary_filename', help='the path to the unary feature file as extracted form extract_cc_features')
parser.add_argument('--pairwise', dest='pairwise_filename', help='the path to the pairwise feature file as extracted form extract_cc_features')
parser.add_argument('--unary-results', dest='unary_result_filename', help='file which includes the unary crossvalidation predictions')
parser.add_argument('--pairwise-results', dest='pairwise_result_filename', help='list of files (comma separated) which include the pairwise crossvalidation predictions')
parser.add_argument('--unary-output', dest='unary_output_filename', help='output file for the unary features')
parser.add_argument('--pairwise-output', dest='pairwise_output_filename', help='output file for the concatenated pairwise features')

args = parser.parse_args()
for name in ['unary_filename', 'pairwise_filename', 'unary_result_filename', 'pairwise_result_filename', 'unary_output_filename', 'pairwise_output_filename']:
    if not getattr(args, name):
        parser.print_help()
        sys.exit(1)

unary = np.genfromtxt(args.unary_filename, delimiter=',')
pw = np.genfromtxt(args.pairwise_filename, delimiter=',')

# need only header
unary = unary[:,0:3]
pw    = pw[:,0:3]

for filename in args.unary_result_filename.split(','):
    unary_results = np.genfromtxt(filename, delimiter=',')
    unary = np.c_[unary, unary_results]

for filename in args.pairwise_result_filename.split(','):
    pw_results = np.genfromtxt(filename, delimiter=',')
    pw = np.c_[pw, pw_results]

np.savetxt(args.unary_output_filename, unary, delimiter=',', fmt='%f')
np.savetxt(args.pairwise_output_filename, pw, delimiter=',', fmt='%f')
