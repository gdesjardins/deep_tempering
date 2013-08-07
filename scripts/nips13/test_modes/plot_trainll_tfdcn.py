import pickle
import os
import numpy
import pylab as pl
import tables
from subprocess import Popen, PIPE

def compute_average(nh2=10, lr_start=0.1, prefix='rbm'):

    cmd = "grep -rl --include='orig.conf' 'nh2 = %i' . | xargs grep 'lr_start = %s' " % (nh2, lr_start)
    print cmd

    x = None
    y = None

    p = os.popen(cmd)
    import pdb; pdb.set_trace()
    for match in p:
        jid = match.split('/')[1]
        rfname = '%s/%s_test_callback.hdf5' % (jid, prefix)
        fp = tables.openFile(rfname)
        if x is None:
            x = fp.root.train_ll.col('n')
            y = fp.root.train_ll.col('train_ll')
        else:
            y = numpy.vstack((y, fp.root.train_ll.col('train_ll')))
    fp.close()

    return [x[::10],
            y.mean(axis=0)[::10],
            y.std(axis=0)[::10]]



for prefix in ['rbm0', 'dbn2']:
    for nh2 in [10]:
        pl.figure()
        for lr in ["0.001", "0.0001", "1e-05"]:
            [x, y, std] = compute_average(nh2, lr, prefix)
            pl.errorbar(x, y, yerr=std, label='%.0e' % numpy.float(lr))
        pl.legend()
        pl.xlabel('nupdates')
        pl.ylabel('train_ll')
        pl.savefig('trainll_%s_nh2=%i.png' % (prefix,nh2))
        pl.close()
