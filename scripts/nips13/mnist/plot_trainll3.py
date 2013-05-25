import pickle
import os
import numpy
import pylab as pl
import tables
from subprocess import Popen, PIPE
from scipy.stats.stats import nanmean, nanstd


def compute_average(nh=10, lr_num=10, lr_denum=1000, prefix='rbm', smoothing=True):
    cmd = "grep -rl --include='orig.conf' 'lr_num = %i$' . |" % lr_num +\
          "xargs grep 'lr_denum = %i$' " % lr_denum
    print cmd

    p = os.popen(cmd)
    numseeds = len([pi for pi in enumerate(p)])
    
    p = os.popen(cmd)
    x = numpy.ones((numseeds, 20)) * numpy.nan
    y = numpy.ones((numseeds, 20)) * numpy.nan

    for i, match in enumerate(p):

        jid = match.split('/')[1]
        rfname = '%s/%s_train_callback.hdf5' % (jid, prefix)
        if not os.path.exists(rfname):
            continue

        fp = tables.openFile(rfname)
        _x = fp.root.train_ll.col('n')
        _y = fp.root.train_ll.col('train_ll')
        _vlogz = fp.root.var_logz.col('var_logz')
        fp.close()

        idx = numpy.where(_vlogz < 10.)[0]
        x[i, idx] = _x[idx]
        y[i, idx] = _y[idx]
    
    print '**** prefix=%s nh=%i lr_num=%s lr_denum=%s ******' % (prefix, nh, lr_num, lr_denum)
    print nanmean(y, axis=0)

    return [nanmean(x, axis=0),
            nanmean(y, axis=0),
            nanstd(y, axis=0)]

lr_config = [(10, 2000),
             (100, 20000),
             (500, 100000),
             (1000, 200000),
             (2000, 400000),
             (4000, 800000)]

nh2 = 500
for prefix in os.sys.argv[1:]:
    assert prefix in ['rbm0', 'dbn1', 'dbn2']

    data = {'x': numpy.zeros((len(lr_config), 20)),
            'y': numpy.ones((len(lr_config), 20)) * -numpy.Inf,
            'std': numpy.zeros((len(lr_config), 20)),
            'label': []}

    pl.figure()
    for i, (lr_num, lr_denum) in enumerate(lr_config):
        label = '(%i,%i)' % (int(lr_num), int(lr_denum))

        [x, y, std] = compute_average(nh2, lr_num, lr_denum, prefix)
        data['x'][i, :len(x)] = x
        data['y'][i, :len(y)] = y
        data['std'][i, :len(std)] = std
        data['label'] += [label]

        pl.errorbar(x, y, yerr=std, label=label)

    pl.legend(loc='lower right')
    pl.xlabel('nupdates')
    pl.ylabel('train_ll')
    pl.ylim((-95,-82))
    pl.savefig('trainll_tdbn_%s_nh2=%i.png' % (prefix,nh2))
    pl.close()

    fp = open('data_%s.pkl' % prefix,'w')
    pickle.dump(data, fp)
    fp.close()

