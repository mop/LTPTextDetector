import sys
if len(sys.argv) < 3:
    print 'usage: conv_pw.py <features> <pw>'

with open(sys.argv[1]) as fp:
    lines = [l.strip().split(',')[:3] for l in fp.readlines()]

label_map = {}
for line in lines:
    label_map['%s-%s' % (line[0],line[1])] = line[2]

def map_to_label(feature, features, label_map):
    img_id = feature[0]
    uid_1 = feature[1]
    uid_2 = feature[2]

    #img_features = [img_feature for img_feature in features if img_feature[0] == img_id]
    #uid1_lbl = [f[2] for f in img_features if f[1] == uid_1]
    #uid2_lbl = [f[2] for f in img_features if f[1] == uid_2]

    #if not uid1_lbl or not uid2_lbl:
    #    print 'zomfg!!!'
    #lbl_1 = uid1_lbl[0]
    #lbl_2 = uid2_lbl[0]

    key1 = '%s-%s' % (img_id, uid_1)
    key2 = '%s-%s' % (img_id, uid_2)
    lbl_1 = label_map[key1]
    lbl_2 = label_map[key2]

    label = -1
    if lbl_1 == '-1' and lbl_2 == '-1':
        label = -1
    if lbl_1 == '-1' and lbl_2 == '1':
        label = 1
    if lbl_1 == '1' and lbl_2 == '-1':
        label = 1

    feature = [str(label)] + feature[3:]
    return feature

with open(sys.argv[2]) as fp:
    pw_lines = [l.strip().split(',') for l in fp.readlines()]
    pw_lines = [','.join(map_to_label(l, lines, label_map)) for l in pw_lines]
    print '\n'.join(pw_lines)
