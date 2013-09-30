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


parser = OptionParser()
parser.add_option('-m', '--model', action='store', type='string', dest='path')
parser.add_option('--nchunks',  action='store', type='int', dest='nchunks', default=10)
parser.add_option('--nchains_parallel', action='store', type='int', dest='nchains_parallel', default=100)
parser.add_option('--nchains_serial', action='store', type='int', dest='nchains_serial', default=1)
parser.add_option('--burnin',  action='store', type='int', dest='burnin', default=1e3)
parser.add_option('--nsamples',  action='store', type='int', dest='nsamples', default=1e4)
parser.add_option('--seed',  action='store', type='int', dest='seed', default=667)
(opts, args) = parser.parse_args()


def cosine_similarity(x, y):
    xx = x / numpy.sqrt(numpy.sum(x**2))
    yy = y / numpy.sqrt(numpy.sum(y**2))
    return numpy.sum(xx * yy)


def set_bsize(model, bsize):
    model.batch_size = bsize
    for rbm in model.rbms:
        rbm.batch_size = bsize
        rbm.init_chains()
        rbm.do_theano()
    model.do_theano()


def reset_chains():
    rng = numpy.random.RandomState(opts.seed)
    for i, rbm in enumerate(model.rbms):
        negv = rng.binomial(n=1, p=0.5, size=(rbm.batch_size, rbm.n_v))
        rbm.neg_v.set_value(negv.astype(floatX))

# load model and retrieve parameters
model = serial.load(opts.path)
rbm = model.rbms[0]

stats = {
        'parallel': numpy.zeros_like(rbm.Wv.get_value()),
        'gibbs': numpy.zeros_like(rbm.Wv.get_value()),
        'dtneg': numpy.zeros_like(rbm.Wv.get_value()),
        }

results = {
    'gibbs': numpy.zeros(opts.nsamples),
    'dtneg': numpy.zeros(opts.nsamples),
}

# build method for computing sufficient statistics
x = T.matrix('x')
fe_x = T.mean(rbm.free_energy_v(x))
compute_stats = theano.function([x], T.grad(fe_x, rbm.Wv))

#### random init ####
rng = numpy.random.RandomState(123)

##
# START WITH PARALLEL IMPLEMENTATION 
##
print 'Computing statistics over parallel chains'
set_bsize(model, opts.nchains_parallel)
for i in xrange(opts.nchunks):
    reset_chains()
    print '\t chunk %i\r' % i,
    sys.stdout.flush()
    for j in xrange(opts.burnin):
        rbm.sample_func()
    stats['parallel'] += compute_stats(rbm.neg_v.get_value())
print ''
stats['parallel'] /= opts.nchunks

##
# NOW THE SERIAL IMPLEMENTATION
##
set_bsize(model, opts.nchains_serial)
reset_chains()
print 'Computing statistics over Gibbs chain'
for i in xrange(opts.nsamples):
    rbm.sample_func()
    _temp = compute_stats(rbm.neg_v.get_value())
    stats['gibbs'] = (_temp + i * stats['gibbs']) / (i + 1)
    results['gibbs'][i] = cosine_similarity(stats['parallel'], stats['gibbs'])
    if i % 100 == 0:
        print '\t step %i / %i \r' % (i, opts.nsamples),
        sys.stdout.flush()
print ''

##
# NOW THE DT IMPLEMENTATION
##
set_bsize(model, opts.nchains_serial)
reset_chains()
print 'Computing statistics over DT chain'
for i in xrange(opts.nsamples):
    model.do_swaps()
    model.do_sample()
    model.batches_seen+=1
    _temp = compute_stats(rbm.neg_v.get_value())
    stats['dtneg'] = (_temp + i * stats['dtneg']) / (i + 1)
    results['dtneg'][i] = mean_cosine_similarity(stats['parallel'], stats['dtneg'])
    if i % 100 == 0:
        print '\t step %i / %i \r' % (i, opts.nsamples),
        sys.stdout.flush()
print ''

pl.plot(results['gibbs'], color='b', label='Gibbs')
pl.plot(results['dtneg'], color='r', label='DT')
pl.legend()
pl.show()

import ipdb; ipdb.set_trace()

