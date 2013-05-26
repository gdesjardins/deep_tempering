import os
import pickle
import numpy
import pylab as pl

ylim = (-95, -85)

######## RBM results ########
fp = open('nips2013_mnist_rbm_exp3_1/data_rbm0.pkl', 'r')
data = pickle.load(fp)
fp.close()

### plot shell of "best" learning rates for each model ###
x  = data['x'][0]
pl.fill_between(x,
        data['y'].min(axis=0),
        data['y'].max(axis=0),
        alpha=0.1,
        color='b')

### now plot the best 2 learning rates for each model ###
i0 = data['y'][:,-1].argsort()[-1]
pl.errorbar(x, data['y'][i0], yerr=data['std'][i0],
            color='b', linestyle='-',
            label='RBM ' + data['label'][i0])
#pl.errorbar(x, data['y'][i1], yerr=data['std'][i1],
            #color='b', linestyle='--',
            #label='RBM ' + data['label'][i1])

######## tDBN-2 results ########
fp = open('nips2013_mnist_tdbn_exp3_1/data_rbm0.pkl', 'r')
data = pickle.load(fp)
fp.close()

### plot shell of "best" learning rates for each model ###
x = data['x'][0]
pl.fill_between(x,
        data['y'].min(axis=0),
        data['y'].max(axis=0),
        alpha=0.1,
        color='g')

### now plot the best 2 learning rates for each model ###
i0 = data['y'][:,-1].argsort()[-1]
pl.errorbar(x, data['y'][i0], yerr=data['std'][i0],
            color='g', linestyle='-',
            label='tDBN-2' + data['label'][i0])
#pl.errorbar(x, data['y'][i1], yerr=data['std'][i1],
            #color='g', linestyle='--',
            #label='tDBN-2' + data['label'][i1])

######## tDBN-3 results ########
fp = open('nips2013_mnist_tdbn_exp3_2/data_rbm0.pkl', 'r')
data = pickle.load(fp)
fp.close()

### plot shell of "best" learning rates for each model ###
x = data['x'][0]
pl.fill_between(x,
        data['y'].min(axis=0),
        data['y'].max(axis=0),
        alpha=0.1,
        color='r')

### now plot the best 2 learning rates for each model ###
i0 = data['y'][:,-1].argsort()[-1]
pl.errorbar(x, data['y'][i0], yerr=data['std'][i0],
            color='r', linestyle='-',
            label='tDBN-3' + data['label'][i0])
#pl.errorbar(x, data['y'][i1], yerr=data['std'][i1],
            #color='r', linestyle='--',
            #label='tDBN-3' + data['label'][i1])

pl.legend(loc='lower right')
pl.savefig('mnist_rbm_vs_tdbn.png')
pl.show()
