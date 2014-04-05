import sys
import numpy as np

if len(sys.argv) < 3:
    print 'usage: make_dataset.py <file> <outfile>'
    sys.exit(1)
input_file, output_file = sys.argv[1:]
print 'using %s' % input_file


data = np.genfromtxt(input_file, delimiter=',')
data = data[:,2:]
print data.shape
data = np.delete(data, [15,16], 1)
print data.shape
np.savetxt(output_file, data, delimiter=',', fmt='%f')
