import sys
import numpy
import pylab as pl
import pickle
from optparse import OptionParser

from theano import function
import theano.tensor as T
import theano

from pylearn2.gui.patch_viewer import make_viewer
from pylearn2.gui.patch_viewer import PatchViewer
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial


parser = OptionParser()
parser.add_option('-m', '--model', action='store', type='string', dest='path')
parser.add_option('--width',  action='store', type='int', dest='width')
parser.add_option('--height', action='store', type='int', dest='height')
parser.add_option('--channels',  action='store', type='int', dest='chans')
parser.add_option('--color', action='store_true',  dest='color', default=False)
parser.add_option('--global',  action='store_false', dest='local',    default=True)
parser.add_option('--preproc', action='store', type='string', dest='preproc')
(opts, args) = parser.parse_args()

nplots = opts.chans
if opts.color:
    assert opts.chans == 3
    nplots = 1

def get_dims(nf):
    num_rows = numpy.floor(numpy.sqrt(nf))
    return (int(num_rows), int(numpy.ceil(nf / num_rows)))

# load model and retrieve parameters
model = serial.load(opts.path)
fov_samples = model.neg_v.get_value()

# store weight matrix as dataset, in case we have to process them
dataset = DenseDesignMatrix(X=fov_samples)
if opts.preproc:
    fp = open(opts.preproc, 'r')
    preproc = pickle.load(fp)
    fp.close()
    print 'Applying inverse pipeline ...'
    preproc.inverse(dataset)
samples = dataset.X

# check for global scaling
if not opts.local:
    samples = samples / numpy.abs(samples).max()

##############
# PLOT FILTERS
##############

import pdb; pdb.set_trace()
viewer = PatchViewer(get_dims(model.batch_size),
                     (opts.height, opts.width),
                     is_color = opts.color,
                     pad=(2,2))

topo_shape = [opts.height, opts.width, opts.chans]
view_converter = DefaultViewConverter(topo_shape)
topo_view = view_converter.design_mat_to_topo_view(samples)

for chan_i in xrange(nplots):

    topo_chan = topo_view if opts.color else topo_view[..., chan_i:chan_i+1]

    for bi in xrange(model.batch_size):
        viewer.add_patch(topo_chan[bi])

    #pl.subplot(1, nplots, chan_i+1)
    #pl.imshow(viewer.image, interpolation=None)
    #pl.axis('off'); pl.title('samples (channel %i)' % chan_i)
    viewer.show()


pl.savefig('weights.png')
pl.show()
