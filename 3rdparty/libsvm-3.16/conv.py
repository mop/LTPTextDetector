import sys

if len(sys.argv) != 2:
    print 'usage: conv.py <filename>'

filename = sys.argv[1]

with open(filename) as fp:
    lines = fp.readlines()
    lines = ['%s %s' % (l.strip().split(',')[0], ' '.join(['%s:%s' % (idx+1, el) for (idx, el) in enumerate(l.strip().split(',')[1:])])) for l in lines]
    print '\n'.join(lines)
