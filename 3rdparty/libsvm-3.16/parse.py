f = open('svs.txt')
txt = [','.join([c.split(':')[1] for c in l.strip().split(' ')[1:]]) for l in f.readlines()]
print '\n'.join(txt)
