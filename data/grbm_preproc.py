"""
GRBM Preprocessor to binarize the (foveated) NORB small dataset.
See "Learning in Markov Random Fields using Tempered Transitions"
Ruslan Salakhutdinov.
"""
import os
import numpy
import theano
import theano.tensor as T

from pylearn2.base import Block
from scipy.io import loadmat
floatX = theano.config.floatX

class GRBMPreprocessor(object):

    def __init__(self):
        base = '%s/norb_small/ruslan_binarized' % os.getenv('PYLEARN2_DATA_PATH')
        self.fname_best = '%s/best.mat' % base
        self.fname_means = '%s/NORB_means.mat' % base
        self.load_params()
        self.do_theano()

    def load_params(self):
        # load GRBM parameters
        params = loadmat(self.fname_best)
        self.Wv = params['vishid_all'].astype(floatX)
        self.hbias = params['hidbiases_all'][0].astype(floatX)
        self.vbias = params['visbiases'][0].astype(floatX)
        self.n_v = len(self.vbias)
        self.n_h = len(self.hbias)
        # load statistics for reconstruction
        means = loadmat(self.fname_means)
        self.zz = numpy.hstack(
                (means['zz1'] * numpy.ones(self.n_v/2, dtype=floatX),
                 means['zz2'] * numpy.ones(self.n_v/2, dtype=floatX)))
        self.vmean = numpy.hstack((means['aa1'], means['aa2']))

    def do_theano(self):
        x = T.matrix()
        h = T.nnet.sigmoid(T.dot(x, self.Wv) + self.hbias)
        self.h_given_x_func = theano.function([x], h)

    def preproc(self, xval):
        return self.h_given_x_func(xval.astype(floatX))

    def reconstruct(self, hval):
        v = numpy.dot(hval, self.Wv.T) + self.vbias
        v *= numpy.sqrt(self.zz)
        return v + self.vmean
