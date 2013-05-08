import pickle
import os
import numpy
import pylab as pl
import tables
from subprocess import Popen, PIPE

def compute_average(nh2=10, lr_start1=0.1, lr_start2=0.1, node='swap0', col=None):

    cmd = "grep -rl --include='orig.conf' 'nh2 = %i' . | xargs grep -l 'lr_start1 = %s' | xargs grep 'lr_start2 = %s' " % (nh2, lr_start1, lr_start2)
    print cmd
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

    for lr1, lr2 in zip(["0.001", "0.0001", "1e-05"], ["0.001", "0.0001", "1e-05"]):
        pl.figure()
        if os.sys.argv[1] == 'swap0':
            [n, s0_mean, s0_std] = compute_average(nh2, lr_start1=lr1, lr_start2=lr2, node='swap0')
            pl.errorbar(n, s0_mean, yerr=s0_std, label='swap0')
        if os.sys.argv[-1] == 'swap1':
            [n, s1_mean, s1_std] = compute_average(nh2, lr_start1=lr1, lr_start2=lr2, node='swap1')
            pl.errorbar(n, s1_mean, yerr=s1_std, label='swap1')
        pl.legend()
        pl.xlabel('nupdates')
        pl.ylabel('Swap ratio')
        pl.ylim((0,1))
        pl.savefig('swaps_nh=%i_lr=%s_mult=1.png' % (nh2, lr1))
        pl.close()

    for lr1, lr2 in zip(["0.001", "0.0001", "1e-05"], ["0.01", "0.001", "0.0001"]):
        pl.figure()
        if os.sys.argv[1] == 'swap0':
            [n, s0_mean, s0_std] = compute_average(nh2, lr_start1=lr1, lr_start2=lr2, node='swap0')
            pl.errorbar(n, s0_mean, yerr=s0_std, label='swap0')
        if os.sys.argv[-1] == 'swap1':
            [n, s1_mean, s1_std] = compute_average(nh2, lr_start1=lr1, lr_start2=lr2, node='swap1')
            pl.errorbar(n, s1_mean, yerr=s1_std, label='swap1')
        pl.legend()
        pl.xlabel('nupdates')
        pl.ylabel('Swap ratio')
        pl.ylim((0,1))
        pl.savefig('swaps_nh=%i_lr=%s_mult=10.png' % (nh2, lr1))
        pl.close()

    for lr1, lr2 in zip(["0.001", "0.0001", "1e-05"], ["0.1", "0.01", "0.001"]):
        pl.figure()
        if os.sys.argv[1] == 'swap0':
            [n, s0_mean, s0_std] = compute_average(nh2, lr_start1=lr1, lr_start2=lr2, node='swap0')
            pl.errorbar(n, s0_mean, yerr=s0_std, label='swap0')
        if os.sys.argv[-1] == 'swap1':
            [n, s1_mean, s1_std] = compute_average(nh2, lr_start1=lr1, lr_start2=lr2, node='swap1')
            pl.errorbar(n, s1_mean, yerr=s1_std, label='swap1')
        pl.legend()
        pl.xlabel('nupdates')
        pl.ylabel('Swap ratio')
        pl.ylim((0,1))
        pl.savefig('swaps_nh=%i_lr=%s_mult=100.png' % (nh2, lr1))
        pl.close()
