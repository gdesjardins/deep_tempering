#!/opt/lisa/os/epd-7.1.2/bin/python
import sys
import copy
import numpy
import pylab as pl
import pickle
import os
from optparse import OptionParser

from theano import function
import theano.tensor as T
import theano
floatX = theano.config.floatX

from pylearn2.gui.patch_viewer import make_viewer
from pylearn2.gui.patch_viewer import PatchViewer
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets import retina
from pylearn2.utils import serial

from deep_tempering.rbm import RBM
from deep_tempering.tempered_dbn import TemperedDBN
from deep_tempering.scripts.likelihood import rbm_tools

parser = OptionParser()
parser.add_option('-m', '--model', action='store', type='string', dest='path')
parser.add_option('--nchunks',  action='store', type='int', dest='nchunks', default=10)
parser.add_option('--nchains', action='store', type='int', dest='nchains', default=100)
parser.add_option('--burnin',  action='store', type='int', dest='burnin', default=1e3)
parser.add_option('--nsamples',  action='store', type='int', dest='nsamples', default=1e4)
parser.add_option('--seed',  action='store', type='int', dest='seed', default=667)
(opts, args) = parser.parse_args()


def mean_cosine_similarity(x, by, breakpoint=False):
    """
    :param x: (n_v, n_h)
    :param by: (bsize, n_v, n_h)
    """
    bsize = by.shape[0]
    norm_x = numpy.sqrt(numpy.sum(x**2))
    norm_by = numpy.sqrt(numpy.apply_over_axes(numpy.sum, by**2, axes=[1,2]))
    temp = x[None, :, :] / norm_x * by / norm_by
    batch_cs = numpy.apply_over_axes(numpy.sum, temp, axes=[1,2])
    if breakpoint:
        import ipdb; ipdb.set_trace()
    return batch_cs.mean(), batch_cs.std()


def set_bsize(model, bsize):
    model.batch_size = bsize
    for rbm in model.rbms:
        rbm.batch_size = bsize
        rbm.init_chains()
        rbm.do_theano()
    model.do_theano()


def reset_chains(seed):
    rng = numpy.random.RandomState(seed)
    for i, rbm in enumerate(model.rbms):
        negv = rng.binomial(n=1, p=0.5, size=(rbm.batch_size, rbm.n_v))
        rbm.neg_v.set_value(negv.astype(floatX))


# load model and retrieve parameters
model = serial.load(opts.path)
rbm = model.rbms[0]

stats = {
        'ref': numpy.zeros_like(rbm.Wv.get_value()),
        'gibbs': numpy.zeros((opts.nchains, rbm.n_v, rbm.n_h)),
        'temp': numpy.zeros((opts.nsamples, rbm.batch_size, rbm.n_v)),
        'dtneg': numpy.zeros((opts.nchains, rbm.n_v, rbm.n_h)),
        }

results = {
    'gibbs_mean': numpy.zeros(opts.nsamples),
    'gibbs_std' : numpy.zeros(opts.nsamples),
    'dtneg_mean': numpy.zeros(opts.nsamples),
    'dtneg_std' : numpy.zeros(opts.nsamples),
}

# build method for computing sufficient statistics
x = T.matrix('x')
fe_x = T.mean(rbm.free_energy_v(x))
batch_compute_stats = theano.function([x], T.grad(fe_x, rbm.Wv))

x = T.matrix('x')
h = rbm.h_given_v(x)
z  = T.shape_padright(x) * rbm.ch.dimshuffle(['x','x',0])
z -= T.shape_padright(x-rbm.cv) * h.dimshuffle([0,'x',1])
compute_stats = theano.function([x], z)

x1 = rbm.neg_v.get_value()
dw1 = batch_compute_stats(x1)
dw2 = compute_stats(x1).mean(axis=0)
assert numpy.allclose(dw1, dw2, atol=1e-6)

#### random init ####
rng = numpy.random.RandomState(123)
set_bsize(model, opts.nchains)

##
# COMPUTE REFERENCE GRADIENT
##
if rbm.n_h < 15:
    print 'Computing exact gradient for reference'
    stats['ref'] = rbm_tools.compute_true_gradient(rbm)
else:
    ##
    print 'Computing reference gradient over parallel chains'
    for i in xrange(opts.nchunks):
        reset_chains(opts.seed * i)
        print '\t chunk %i\r' % i,
        sys.stdout.flush()
        for j in xrange(opts.burnin):
            rbm.sample_func()
        stats['ref'] += batch_compute_stats(rbm.neg_v.get_value())
    print ''
    stats['ref'] /= opts.nchunks


##
# NOW THE SERIAL IMPLEMENTATION
##
reset_chains(opts.seed)
print 'Computing statistics over Gibbs chain'
for i in xrange(opts.nsamples):
    rbm.sample_func()
    _temp = compute_stats(rbm.neg_v.get_value())
    stats['gibbs'] = (_temp + i * stats['gibbs']) / (i + 1)
    cs_mean, cs_std = mean_cosine_similarity(stats['ref'], stats['gibbs'])
    results['gibbs_mean'][i] = cs_mean
    results['gibbs_std'][i] = cs_std
    if i % 100 == 0:
        print '\t step %i / %i \r' % (i, opts.nsamples),
        sys.stdout.flush()
print ''


##
# NOW THE DT IMPLEMENTATION
##
reset_chains(opts.seed)
print 'Computing statistics over DT chain'
for i in xrange(opts.nsamples):
    model.do_swaps()
    model.do_sample()
    model.batches_seen+=1
    _temp = compute_stats(rbm.neg_v.get_value())
    stats['dtneg'] = (_temp + i * stats['dtneg']) / (i + 1)
    if i== 500:
        cs_mean, cs_std = mean_cosine_similarity(stats['ref'], stats['dtneg'])
    else:
        cs_mean, cs_std = mean_cosine_similarity(stats['ref'], stats['dtneg'])
    results['dtneg_mean'][i] = cs_mean
    results['dtneg_std'][i] = cs_std
    if i % 100 == 0:
        print '\t step %i / %i \r' % (i, opts.nsamples),
        sys.stdout.flush()
print ''


print 'Swap ratios:', model.swap_ratios
x = numpy.arange(0, opts.nsamples)
y = results['gibbs_mean']
std = results['gibbs_std']
pl.plot(x, y, color='b', label='Gibbs')
pl.fill_between(x, y-std, y+std, alpha=0.1, color='b')

x = numpy.arange(0, opts.nsamples)
y = results['dtneg_mean']
std = results['dtneg_std']
pl.plot(x, y, color='r', label='DT')
pl.fill_between(x, y-std, y+std, alpha=0.1, color='r')

pl.legend()
pl.show()

