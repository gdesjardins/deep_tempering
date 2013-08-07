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

def _softplus(x):
    y = numpy.log(1. + numpy.exp(x))
    y[x > 30.] = x[x > 30.]
    y[x < 30.].fill(0)
    return y

def _get_mask_half(im_shape, left=True):
    mask = numpy.zeros(im_shape)
    if left:
        mask[:, :im_shape[1]/2] = 1.
    else:
        mask[:, im_shape[1]/2:] = 1.
    return mask.flatten().astype(numpy.bool)


def _fe_cond(rbm, sample, mask, over_hiddens=True):
    """
    Compute free-energy of rbm.neg_v under the RBM distribution,
    conditioned on the hidden samples rbm.neg_h[:, hmask].
    """
    # retrieve numerical values from shared variables.
    Wv = rbm.Wv.get_value()
    vbias = rbm.vbias.get_value()
    hbias = rbm.hbias.get_value()
    neg_v = rbm.neg_v.get_value()
    neg_h = rbm.neg_h.get_value()
    cv = rbm.cv.get_value()
    ch = rbm.ch.get_value()

    if not over_hiddens:
        (neg_v, cv, ch, vbias, hbias, Wv) = \
            (neg_h, ch, cv, hbias, vbias, Wv.T)

    fe  = 0.
    # energy contribution of visible units being conditioned on
    fe -= numpy.dot(neg_v[:, mask], vbias[mask])
    fe -= numpy.sum(numpy.dot(neg_v[:, mask], Wv[mask, :]) * sample, axis=1)
    # "normal" free-energy over h's, after marginalizing subset of v's.
    fe -= numpy.dot(sample, hbias)
    fe += numpy.sum(numpy.dot(sample, Wv[~mask, :].T) * cv[~mask], axis=1)
    v_input  = numpy.dot(sample - ch, Wv[~mask, :].T) + vbias[~mask]
    fe -= numpy.sum(_softplus(v_input), axis=1)
    return fe


class GibbsSampler():
    """
    Performs simple Gibbs sampling in the lower layer RBM.
    """

    def __init__(self, model, mask):
        model.do_theano()
        for rbm in model.rbms:
            rbm.do_theano()
        self.model = model
        self.mask = mask

    def clamp(self, negv, x):
        buff = negv.get_value()
        buff[:, self.mask] = x[:, self.mask]
        negv.set_value(buff)

    def run(self, x, n_steps=1):
        rbm0 = self.model.rbms[0]
        self.clamp(rbm0.neg_v, x)
        for i in xrange(n_steps):
            # run one Gibbs step at each layer
            self.model.do_sample()
            self.clamp(rbm0.neg_v, x)


class DTSampler():
    """
    Performs a conditional version of DTNegPhase.
    """

    def __init__(self, model, mask):
        model.do_theano()
        for rbm in model.rbms:
            rbm.do_theano()
        self.model = model
        self.v0_mask = mask
        self.swap_ratios = [0 for i in range(model.depth-1)]
        self.count = 0
        self.rng = numpy.random.RandomState(1234)

    def clamp(self, negv, x):
        buff = negv.get_value()
        buff[:, self.v0_mask] = x[:, self.v0_mask]
        negv.set_value(buff)

    def run(self, x, n_steps=1):
        # run greedy inference process
        qs = self.model.inference_func(x)
        # set visibles to mean hidden activations of prev. layer
        self.set_visibles(qs)

        for i in xrange(n_steps):
            # run one Gibbs step at each layer
            self.model.do_sample()
            self.model.do_swaps()
            # keep subset of visible units clamped
            self.clamp(self.model.rbms[0].neg_v, x)
            self.count += 1

        print 'swap_ratio = ', self.model.swap_ratios

    def set_visibles(self, qs):
        """
        For each RBM in the stack, configure visible sample to:
        \mathbb{E}_{q_{i-1}} p_h (h_i \mid v_i)
        """
        for rbm, q in zip(self.model.rbms, qs):
            rbm.neg_v.set_value(q)


class ConditionalDTSampler():
    """
    Performs a conditional version of DTNegPhase.
    """

    def __init__(self, model, mask, cond_v_only=False):
        model.do_theano()
        for rbm in model.rbms:
            rbm.do_theano()
        self.model = model
        self.v0_mask = mask
        self.swap_ratios = [0 for i in range(model.depth-1)]
        self.count = 0
        self.cond_v_only = cond_v_only
        self.rng = numpy.random.RandomState(1234)

    def clamp(self, negv, x):
        buff = negv.get_value()
        buff[:, self.v0_mask] = x[:, self.v0_mask]
        negv.set_value(buff)

    def run(self, x, n_steps=1):
        # run greedy inference process
        qs = self.model.inference_func(x)
        # set visibles to mean hidden activations of prev. layer
        self.set_visibles(qs)

        for i in xrange(n_steps):
            # run one Gibbs step at each layer
            self.model.do_sample()
            # then perform conditional swaps
            start_idx = 0 if (self.model.depth == 2 or self.count % 2 == 0) else 1
            for j in xrange(start_idx, self.model.depth, 2):
                self.do_conditional_swap(j)
            # keep subset of visible units clamped
            self.clamp(self.model.rbms[0].neg_v, x)
            self.count += 1

        print 'swap_ratio = ', self.swap_ratios

    def set_visibles(self, qs):
        """
        For each RBM in the stack, configure visible sample to:
        \mathbb{E}_{q_{i-1}} p_h (h_i \mid v_i)
        """
        for rbm, q in zip(self.model.rbms, qs):
            rbm.neg_v.set_value(q)

    def do_conditional_swap(self, i):
        """ Perform swaps between samples of i-th and (i+1)-th RBM """
        assert i+1 < len(self.model.rbms)
        rbm_i   = self.model.rbms[i]
        rbm_ip1 = self.model.rbms[i+1]

        # We are proposing a swap between:
        # hi ~ p_i(h_i | v_i[vi_mask]) andi
        # v_{i+1} ~ p_{i+1}(v_{i+1} | h_{i+1}[hip1_mask])
        vi   = rbm_i.neg_v.get_value()
        hi   = rbm_i.neg_h.get_value()
        vip1 = rbm_ip1.neg_v.get_value()

        if self.cond_v_only:
            hip1_mask = numpy.zeros(rbm_ip1.n_h).astype(numpy.bool)
            if i == 0:
                vi_mask = self.v0_mask
            else:
                vi_mask = numpy.zeros(rbm_i.n_v).astype(numpy.bool)
        else:
            hip1_mask = numpy.ones(rbm_ip1.n_h).astype(numpy.bool)
            if i == 0:
                vi_mask = self.v0_mask
            else:
                vi_mask = numpy.ones(rbm_i.n_v).astype(numpy.bool)

        # compute log-probability of new / old configuration
        logp_old1 = _fe_cond(rbm_i, hi, vi_mask, over_hiddens=True)
        logp_old2 = _fe_cond(rbm_ip1, vip1, hip1_mask, over_hiddens=False)
        logp_new1 = _fe_cond(rbm_i, vip1, vi_mask, over_hiddens=True)
        logp_new2 = _fe_cond(rbm_ip1, hi, hip1_mask, over_hiddens=False)

        # compute swap probability
        logr = logp_new1 + logp_new2 - logp_old1 - logp_old2
        r = numpy.minimum(1, numpy.exp(logr))
        swap = self.rng.binomial(n=1, p=r, size=(len(r),)).astype(floatX)

        ### Perform swap h_i <=> v_{i+1} direction.
        vi[swap == 1] = rbm_i.sample_v_given_h_func(hi)[swap == 1]
        vip1[swap == 1] = hi[swap == 1] 

        # Update shared variables with new state.
        rbm_i.neg_v.set_value(vi)
        rbm_ip1.neg_v.set_value(vip1)

        self.swap_ratios[i] = (self.count * self.swap_ratios[i] + swap.mean()) / (self.count + 1.)


class DBNSampler():
    """
    Performs a conditional version of DTNegPhase.
    """

    def __init__(self, model, mask, redo_inference=False):
        model.do_theano()
        for rbm in model.rbms:
            rbm.do_theano()
        self.model = model
        self.v0_mask = mask
        self.count = 0
        self.redo_inference = redo_inference
        self.rng = numpy.random.RandomState(1234)

    def clamp(self, negv, x):
        buff = negv.get_value()
        buff[:, self.v0_mask] = x[:, self.v0_mask]
        negv.set_value(buff)

    def propdown(self):
        negv = self.model.rbms[-1].neg_v.get_value()
        for rbm in self.model.rbms[:-1][::-1]:
            negv = rbm.sample_v_given_h_func(negv)
            rbm.neg_v.set_value(negv)

    def run(self, x, n_steps=1):
        # run greedy inference process
        qs = self.model.inference_func(x)
        # set visibles to mean hidden activations of prev. layer
        self.set_visibles(qs)

        for i in xrange(n_steps):
            self.model.rbms[-1].sample_func()
            if self.redo_inference:
                self.propdown()
                self.clamp(self.model.rbms[0].neg_v, x)
                qs = self.model.inference_func(self.model.rbms[0].neg_v.get_value())
                self.set_visibles(qs)

        self.propdown()
        self.clamp(self.model.rbms[0].neg_v, x)
        self.count += 1

    def set_visibles(self, qs):
        """
        For each RBM in the stack, configure visible sample to:
        \mathbb{E}_{q_{i-1}} p_h (h_i \mid v_i)
        """
        for rbm, q in zip(self.model.rbms, qs):
            rbm.neg_v.set_value(q)


def show(x, xhat):
    temp = numpy.zeros((2*len(x), x.shape[1]))
    temp[::2] = x
    temp[1::2] = xhat
    make_viewer(temp, (10,20), (28,28)).show()

def print_stats(x, xhat):
    print 'Mean error:', numpy.sum((x - xhat)**2, axis=1).mean()



class InPainting():

    options = {
        'batch_size': 100,
        'fill_value': 0.5,
        'sample_steps': 100,
    }

    def __init__(self, model, dataset, sampler, mask, options=None):

        self.model = model
        self.dataset = dataset
        self.sampler = sampler
        self.vmask = mask
        self.options.update(options if options else {})
        self.im_shape = self.dataset.view_shape()

    def run(self):
        x = self.dataset.get_batch_design(self.options['batch_size'])
        mask_x = copy.copy(x)
        mask_x[:, ~self.vmask] = 0.5

        self.sampler.run(mask_x, n_steps = self.options['sample_steps'])
        xhat = self.model.rbms[0].neg_v.get_value()

        print_stats(x, xhat)
        show(x, xhat)


if __name__ == '__main__':
    from pylearn2.config import yaml_parse 
    assert len(os.sys.argv) > 1
    fname = os.sys.argv[1]

    fp = open(fname)
    inpainter = yaml_parse.load(fp)
    fp.close()

    inpainter.run()



