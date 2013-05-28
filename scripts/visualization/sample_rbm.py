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
model.init_chains()
model.do_theano()

#### random init ####
rng = numpy.random.RandomState(123)
bsize = model.batch_size
samples = numpy.zeros((opts.nsamples, bsize, model.n_v))

def reset_chains():
    vbias = sigm(model.vbias.get_value())
    negv = rng.binomial(n=1, p=vbias, size=(bsize, model.n_v))
    model.neg_v.set_value(negv.astype(floatX))

reset_chains()
for i in xrange(opts.burnin):
    model.sample_func()

for i in xrange(opts.nsamples):
    for j in xrange(opts.skip):
        model.sample_func()
    samples[i, :, :] = model.neg_ev.get_value()

if opts.norb:
    from deep_tempering.data import grbm_preproc
    grbm = grbm_preproc.GRBMPreprocessor()
    _temp = grbm.reconstruct(samples.reshape(-1, model.n_v))
    del samples
    samples = _temp.reshape(opts.nsamples, bsize, _temp.shape[1])

viewer = PatchViewer((opts.nsamples, opts.batches_per_image),
                     (opts.height, opts.width),
                      is_color = opts.color,
                      pad=(10,10))

mbsize = opts.batches_per_image
for kbase in xrange(0, bsize, mbsize):
    for i in xrange(opts.nsamples):

        bidx = slice(kbase, kbase+mbsize)
        if opts.rings:
            b_sample = retina.decode(samples[i, bidx], (opts.height, opts.width, opts.chans), opts.rings)
        else:
            b_sample = samples[i, bidx].reshape(bsize, opts.height, opts.width, opts.chans)

        for k in xrange(mbsize):
            viewer.add_patch(b_sample[k,:,:,0])

    pl.imshow(viewer.image)
    pl.savefig('samples_batch%i.png' % kbase)
    viewer.clear()
