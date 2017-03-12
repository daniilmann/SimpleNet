# -*- encoding: utf-8 -*-

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def classification(learn, valid, test):

    ly, lp = learn[0].squeeze(), learn[1].squeeze()
    vy, vp = valid[0].squeeze(), valid[1].squeeze()
    ty, tp = test[0].squeeze(), test[1].squeeze()

    lr, vr, tr = None, None, None

    if len(np.unique(lp)) != 2:
        lp = np.round(lp)

    if len(np.unique(vp)) != 2:
        vp = np.round(vp)

    if len(np.unique(tp)) != 2:
        tp = np.round(tp)

    if len(learn) == 3:
        lr = learn[2]

    if len(valid) == 3:
        vr = valid[2]

    if len(test) == 3:
        tr = test[2]

    print classification_report(ty, tp)
    print confusion_matrix(ty, tp)
    l = ['%.4s  ||  %.4s' % (float(sum(ly == np.sign(learn[1].squeeze()))) / len(ly), 0)]# float(sum(ly == lp)) / (len(ly) - sum(lp == 0)))]
    v = ['%.4s  ||  %.4s' % (float(sum(vy == np.sign(valid[1].squeeze()))) / len(vy), 0)]#float(sum(vy == vp)) / (len(vy) - sum(vp == 0)))]
    t = ['%.4s  ||  %.4s' % (float(sum(ty == np.sign(test[1].squeeze()))) / len(ty), 0)]#float(sum(ty == tp)) / (len(ty) - sum(tp == 0)))]
    if lr is not None:
        l.append('%.5s  ||  %+.3f  ||  %+.5f' % (sum(ly * lr), sum(lp * lr), sum(lp * lr) / len(lr)))
    print 'learn: ', '  ||  '.join(l)
    if vr is not None:
        v.append('%.5s  ||  %+.3f  ||  %+.5f' % (sum(vy * vr), sum(vp * vr), sum(vp * vr) / len(vr)))
    print 'valid: ', '  ||  '.join(v)
    if tr is not None:
        t.append('%.5s  ||  %+.3f  ||  %+.5f' % (sum(ty * tr), sum(tp * tr), sum(tp * tr) / len(tr)))
    print 'test:  ', '  ||  '.join(t)