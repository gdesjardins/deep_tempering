import os
import pickle
import numpy
import pylab as pl

dirs = os.sys.argv[1:]
y = numpy.zeros((len(dirs), 20))
clrs = ['g','r','b']

for i, diri in enumerate(dirs):
    fp = open('%s/data_rbm0.pkl' % diri, 'r')
    data = pickle.load(fp)
    fp.close()

    ### plot shell of "best" learning rates for each model ###
    x = data['x'][0]
    y = data['y'].max(axis=0)
    pl.fill_between(x, -95, y, label='%s'%diri, alpha=0.1, color=clrs[i])

    ### now plot the best 2 learning rates for each model ###
    (i0, i1) = data['y'][:,-1].argsort()[-2:]
    pl.errorbar(x, data['y'][i0], yerr=data['std'][i0],
                fmt=clrs[i]+'-',  label=data['label'][i0])
    pl.errorbar(x, data['y'][i1], yerr=data['std'][i1],
                fmt=clrs[i]+'--', label=data['label'][i1])

pl.legend()
pl.show()
