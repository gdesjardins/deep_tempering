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
parser.add_option('--width',  action='store', type='int', dest='width')
parser.add_option('--height', action='store', type='int', dest='height')
parser.add_option('--channels',  action='store', type='int', dest='chans')
parser.add_option('--color', action='store_true',  dest='color', default=False)
parser.add_option('--rings',  action='store', type='string', dest='rings', default=None)
parser.add_option('--nsamples',  action='store', type='int', dest='nsamples')
parser.add_option('--burnin',  action='store', type='int', dest='burnin')
parser.add_option('--skip',  action='store', type='int', dest='skip')
parser.add_option('--norb',  action='store_true', dest='norb')
parser.add_option('--batches_per_image',  action='store', type='int', dest='batches_per_image', default=10)
parser.add_option('--bsize',  action='store', type='int', dest='bsize', default=None)
parser.add_option('--savelast',  action='store_true', dest='savelast', default=False)
parser.add_option('--init_from',  action='store', type='string', dest='init_from', default=None)
(opts, args) = parser.parse_args()

def get_dims(nf):
    num_rows = numpy.floor(numpy.sqrt(nf))
    return (int(num_rows), int(numpy.ceil(nf / num_rows)))

def sigm(x): return 1./(1. + numpy.exp(-x))

nplots = opts.chans
if opts.color:
    assert opts.chans == 3
    nplots = 1
if opts.rings:
    opts.rings = eval(opts.rings)
assert opts.color is False

topo_shape = [opts.height, opts.width, opts.chans]
viewconv = DefaultViewConverter(topo_shape)
viewdims = slice(0, None) if opts.color else 0

# load model and retrieve parameters
model = serial.load(opts.path)

# optionally change minibatch size
if opts.bsize:
    model.batch_size = opts.bsize
bsize = model.batch_size

model.init_chains()
model.do_theano()

#### random init ####
rng = numpy.random.RandomState(123)
samples = numpy.zeros((opts.nsamples, bsize, model.n_v), dtype='uint8')

def reset_chains():
    """
    Reset model's Markov chains to random or to test examples from a given distribution.
    """
    if opts.init_from == 'mnist':
        from pylearn2.datasets import mnist
        data = mnist.MNIST('test', center=False, one_hot=True, binarize='sample')
        idx = rng.permutation(numpy.arange(len(data.X)))[:bsize]
        model.neg_v.set_value(data.X[idx])
    elif opts.init_from == 'random':
        negv = rng.binomial(n=1, p=0.5, size=(bsize, model.n_v))
        model.neg_v.set_value(negv.astype(floatX))
    else:
        raise ValueError('Wrong value for parameter `init_from`')

reset_chains()
for i in xrange(opts.burnin):
    model.sample_func()

for i in xrange(opts.nsamples):
    for j in xrange(opts.skip):
        model.sample_func()
    samples_i = model.neg_v.get_value()
    samples[i] = (samples_i * 255).astype('uint8')
    print 'sampling: %i/%i\r' % (i, opts.nsamples),

if opts.norb:
    from deep_tempering.data import grbm_preproc
    grbm = grbm_preproc.GRBMPreprocessor()
    _temp = grbm.reconstruct(samples.reshape(-1, model.n_v))
    del samples
    samples = _temp.reshape(opts.nsamples, bsize, _temp.shape[1])

if not opts.savelast:
    numpy.save('rbm_samples.npy', samples)
    numpy.save('rbm_samples_batch0.npy', samples[:,0,:].reshape(-1, 1, opts.width, opts.height))
else:
    numpy.save('rbm_samples_last.npy', samples[-1,:,:])
    numpy.save('rbm_samples_mean.npy', samples[-1].mean(axis=0).reshape(1, opts.width, opts.height))
