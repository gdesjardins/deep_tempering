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
parser.add_option('--layer',  action='store', type='int', dest='layer', default=-1)
parser.add_option('--skip',  action='store', type='int', dest='skip')
parser.add_option('--norb',  action='store_true', dest='norb')
parser.add_option('--batches_per_image',  action='store', type='int', dest='batches_per_image', default=10)
(opts, args) = parser.parse_args()

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
assert opts.layer < len(model.rbms)
layer = len(model.rbms) - 1 if opts.layer == -1 else opts.layer
model.do_theano()
for rbm in model.rbms:
    rbm.init_chains()
    rbm.do_theano()

#### random init ####
rng = numpy.random.RandomState(123)
bsize = model.rbms[0].batch_size
nv = model.rbms[0].n_v
samples = numpy.zeros((opts.nsamples, bsize, nv))

def reset_chains():
    for rbm in model.rbms:
        vbias = sigm(rbm.vbias.get_value())
        negv = rng.binomial(n=1, p=vbias,
                size=(bsize, rbm.n_v))
        rbm.neg_v.set_value(negv.astype(floatX))

reset_chains()
for i in xrange(opts.burnin):
    model.do_sample()
    print 'Burn-in: %i/%i\r' % (i, opts.burnin),
print ''

for i in xrange(opts.nsamples):
    for j in xrange(opts.skip):
        model.do_swaps()
        model.do_sample()
    samples[i] = model.rbms[0].neg_ev.get_value()
    print 'sampling: %i/%i\r' % (i, opts.nsamples),
print ''

if opts.norb:
    from deep_tempering.data import grbm_preproc
    grbm = grbm_preproc.GRBMPreprocessor()
    _temp = grbm.reconstruct(samples.reshape(-1, nv))
    samples = _temp.reshape(samples.shape)


viewer = PatchViewer((opts.nsamples, opts.batches_per_image),
                     (opts.height, opts.width),
                      is_color = opts.color,
                      pad=(10,10))

mbsize = opts.batches_per_image
for bidx in xrange(0, bsize, mbsize):

    batch = samples[:, bidx:bidx + mbsize, :]

    for i in xrange(opts.nsamples):

        batch_at_ith_sample = batch[i, :]
        if opts.rings:
            b_sample = retina.decode(
                    batch_at_ith_sample,
                    (opts.height, opts.width, opts.chans),
                    opts.rings)
        else:
            b_sample = batch_at_ith_sample.reshape(mbsize, opts.height, opts.width, opts.chans)

        for mbidx in xrange(len(b_sample)):
            viewer.add_patch(b_sample[mbidx,:,:,0])

    pl.imshow(viewer.image)
    pl.savefig('samples_batch%i.png' % bidx)
    viewer.clear()
