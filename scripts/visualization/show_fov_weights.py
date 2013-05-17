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
parser.add_option('--rings',  action='store', type='string', dest='rings')
parser.add_option('--color', action='store_true',  dest='color', default=False)
parser.add_option('--top', action='store', type='int', dest='top', default=5)
parser.add_option('--layer', action='store',  type='int', dest='layer', default=0)
(opts, args) = parser.parse_args()

nplots = opts.chans
if opts.color:
    assert opts.chans == 3
    nplots = 1

def get_dims(nf):
    num_rows = numpy.floor(numpy.sqrt(nf))
    return (int(num_rows), int(numpy.ceil(nf / num_rows)))

topo_shape = [opts.height, opts.width, opts.chans]
viewconv = DefaultViewConverter(topo_shape)
viewdims = slice(0, None) if opts.color else 0

# load model and retrieve parameters
model = serial.load(opts.path)
if isinstance(model, TemperedDBN):
    rbm = model.rbms[opts.layer]
else:
    rbm = model

wv = rbm.Wv.get_value().T
if opts.rings:
    rings = eval(opts.rings)
    wv = retina.decode(wv, (opts.height, opts.width, opts.chans), rings)


### Build channels for individual channels ###
chans_viewer = []
for chani in range(opts.chans):

    patch_viewer = PatchViewer(
            get_dims(len(wv)),
            (opts.height, opts.width),
            is_color = opts.color,
            pad=(2,2))
    chans_viewer += [patch_viewer]

    for i in range(len(wv)):
        patch_viewer.add_patch(wv[i,:,:,chani])


main_viewer = PatchViewer((1,opts.chans),
                          (chans_viewer[0].image.shape[0],
                           chans_viewer[0].image.shape[1]),
                          is_color = opts.color,
                          pad=(10,10))
for chan_viewer in chans_viewer:
    main_viewer.add_patch(chan_viewer.image[:,:,viewdims] - 0.5)
main_viewer.show()
