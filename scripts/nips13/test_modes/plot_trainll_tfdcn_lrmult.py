import pickle
import os
import numpy
import pylab as pl
import tables
from subprocess import Popen, PIPE

def compute_average(lr_start1=0.1, lr_start2=1., nh2=10):

    cmd = "grep -rl --include='orig.conf' 'lr_start1 = %s' . | xargs grep -l 'lr_start2 = %s' | xargs grep 'nh2 = %i' " % (lr_start1, lr_start2, nh2)

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

    return [x[::10],
            y.mean(axis=0)[::10],
            y.std(axis=0)[::10]]

for nh2 in [10, 50]:

    pl.figure()
    for lr1, lr2 in zip(["0.001", "0.0001", "1e-05"], ["0.001", "0.0001", "1e-05"]):
        [x, y, std] = compute_average(lr_start1=lr1, lr_start2=lr2, nh2=10)
        pl.errorbar(x, y, yerr=std, label='%.0e' % numpy.float(lr1))
    pl.legend()
    pl.xlabel('nupdates')
    pl.ylabel('train_ll')
    pl.savefig('trainll_nh2=%i_mult=1.png' % nh2)
    pl.close()


    pl.figure()
    for lr1, lr2 in zip(["0.001", "0.0001", "1e-05"], ["0.01", "0.001", "0.0001"]):
        [x, y, std] = compute_average(lr_start1=lr1, lr_start2=lr2, nh2=10)
        pl.errorbar(x, y, yerr=std, label='%.0e' % numpy.float(lr1))
    pl.legend()
    pl.xlabel('nupdates')
    pl.ylabel('train_ll')
    pl.savefig('trainll_nh2=%i_mult=10.png' % nh2)
    pl.close()

    pl.figure()
    for lr1, lr2 in zip(["0.001", "0.0001", "1e-05"], ["0.1", "0.01", "0.001"]):
        [x, y, std] = compute_average(lr_start1=lr1, lr_start2=lr2, nh2=10)
        pl.errorbar(x, y, yerr=std, label='%.0e' % numpy.float(lr1))
    pl.legend()
    pl.xlabel('nupdates')
    pl.ylabel('train_ll')
    pl.savefig('trainll_nh2=%i_mult=100.png' % nh2)
    pl.close()
