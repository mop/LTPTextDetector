p = 0.93
r = 0.92

positives = 2051
negatives = 8668

tp = r * positives
fn = positives - tp

fp = 1/(p/tp) - tp

print 1.0 - (fp + fn) / (positives + negatives)
