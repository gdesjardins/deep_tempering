import pickle
import os
import numpy
import pylab as pl
import tables
from subprocess import Popen, PIPE

def compute_average(nh1=10, lr_start=0.1):

    cmd = "grep -rl --include='orig.conf' 'nh1 = %i' . | xargs grep 'lr_start = %s' " % (nh1, lr_start)

    x = None
    y = None

    p = os.popen(cmd)
    for match in p:
        jid = match.split('/')[1]
        rfname = '%s/exactll.hdf5' % jid
        fp = tables.openFile(rfname)
        if x is None:
            x = fp.root.train_ll.col('n')
            y = fp.root.train_ll.col('train_ll')
        else:
            y = numpy.vstack((y, fp.root.train_ll.col('train_ll')))
    fp.close()

    return [x, y.mean(axis=0), y.std(axis=0)]

for nh1 in [10, 20]:

    pl.figure()
    for lr in ["0.001", "0.0001", "1e-05"]:
        [x, y, std] = compute_average(nh1, lr)
        pl.errorbar(x, y, yerr=std, label='%.0e' % numpy.float(lr))
    pl.legend()
    pl.xlabel('nupdates')
    pl.ylabel('train_ll')
    pl.savefig('trainll_rbm_nh1=%i.png' % nh1)
    pl.close()
