import pickle
import os
import numpy
import pylab as pl
import tables
from subprocess import Popen, PIPE

def compute_average(nh2=10, lr_start=0.1, node='swap0', col=None):

    cmd = "grep -rl --include='orig.conf' 'nh2 = %i' . | xargs grep 'lr_start = %s' " % (nh2, lr_start)
    col = node if col is None else col

    x = None
    y = None

    p = os.popen(cmd)
    for match in p:
        jid = match.split('/')[1]
        rfname = '%s/swap_callback.hdf5' % jid
        fp = tables.openFile(rfname)
        node = getattr(fp.root, col)
        if x is None:
            x = node.col('n')
            y = node.col(col)
        else:
            y = numpy.vstack((y, node.col(col)))
    fp.close()

    return [x[::10],
            y.mean(axis=0)[::10],
            y.std(axis=0)[::10]]

assert len(os.sys.argv[1]) > 1
for nh2 in [10, 50]:
    for lr in ["0.001", "0.0001", "1e-05"]:
        pl.figure()
        if os.sys.argv[1] == 'swap0':
            [n, s0_mean, s0_std] = compute_average(nh2, lr, node='swap0')
            pl.errorbar(n, s0_mean, yerr=s0_std, label='swap0')
        if os.sys.argv[-1] == 'swap1':
            [n, s1_mean, s1_std] = compute_average(nh2, lr, node='swap1')
            pl.errorbar(n, s1_mean, yerr=s1_std, label='swap1')
        pl.legend()
        pl.xlabel('nupdates')
        pl.ylabel('Swap ratio')
        pl.ylim((0,1))
        pl.savefig('swaps_nh=%i_lr=%s.png' % (nh2, lr))
        pl.close()
