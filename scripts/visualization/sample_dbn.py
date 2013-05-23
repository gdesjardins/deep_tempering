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
assert opts.chans == 1

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
samples = numpy.zeros((bsize, opts.nsamples, model.rbms[0].n_v))

def reset_chains():
    for rbm in model.rbms:
        vbias = sigm(rbm.vbias.get_value())
        negv = rng.binomial(n=1, p=vbias,
                size=(bsize, rbm.n_v))
        rbm.neg_v.set_value(negv.astype(floatX))

def sample_layer(i):
    for k in xrange(opts.skip):
        model.rbms[i].sample_func()
    return model.rbms[i].neg_ev.get_value()

def propdown_from(i, negv):
    assert i > 0
    negv = model.rbms[i-1].sample_v_given_h_func(negv)
    model.rbms[i-1].neg_v.set_value(negv)

reset_chains()
for i in xrange(opts.burnin):
    sample_layer(layer)

for i in xrange(opts.nsamples):
    for l in numpy.arange(0, layer+1)[::-1]:
        negv = sample_layer(l)
        if l > 0:
            propdown_from(l, negv)
    samples[:,i,:] = negv

viewer = PatchViewer(get_dims(opts.nsamples),
                     (opts.height, opts.width),
                      is_color = opts.color,
                      pad=(10,10))

for k in xrange(bsize):

    if opts.rings:
        b_sample = retina.decode(samples[k], (opts.height, opts.width, opts.chans), opts.rings)
    else:
        b_sample = samples[k].reshape(opts.nsamples, opts.height, opts.width, opts.chans)

    for i in xrange(opts.nsamples):
        viewer.add_patch(b_sample[i,:,:,0])

    pl.imshow(viewer.image)
    pl.savefig('samples_batch%i.png' % k)
    viewer.clear()
