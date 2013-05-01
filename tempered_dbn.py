import numpy
import copy
import time
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import function, shared

from pylearn2.training_algorithms import default
from pylearn2.utils import serial
from pylearn2.base import Block
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace

from utils import cost as costmod
from utils import rbm_utils
from utils import sharedX, floatX, npy_floatX

class TemperedDBN(Model, Block):

    def __init__(self, rbms=None, max_updates=1e6, **kwargs):
        # dump initialization parameters to object
        for (k,v) in locals().iteritems():
            if k!='self': setattr(self,k,v)

        # validate that RBMs have the same number of units.
        for (rbm1, rbm2) in zip(rbms[:-1], rbms[1:]):
            assert rbm1.n_h == rbm2.n_v
            assert rbm1.batch_size == rbm2.batch_size
        self.rbms = rbms
        self.depth = len(rbms)

        # configure input-space (necessary evil)
        self.input_space = VectorSpace(self.rbms[0].n_v)
        self.output_space = VectorSpace(self.rbms[-1].n_h)

        self.batches_seen = 0  # incremented on every batch
        self.examples_seen = 0 # incremented on every training example
        self.batch_size = self.rbms[0].batch_size
        self.cpu_time = 0
        self.do_theano()

    def do_theano(self):
        self.build_swap_funcs()

    def build_swap_funcs(self):
        self.swap_funcs = []
        self.swap_ratios = []
        for idx, (rbm1, rbm2) in enumerate(zip(self.rbms[:-1], self.rbms[1:])):
            #logp_old1 = -rbm1.free_energy_h(rbm1.neg_h)
            rbm1_negh = rbm1.sample_h_given_v(rbm1.neg_v)
            logp_old1 = -rbm1.free_energy_h(rbm1_negh)
            logp_old2 = -rbm2.free_energy_v(rbm2.neg_v)
            logp_new1 = -rbm1.free_energy_h(rbm2.neg_v)
            logp_new2 = -rbm2.free_energy_v(rbm1_negh)
            logr = logp_new1 + logp_new2 - logp_old1 - logp_old2
            r = T.minimum(1, T.exp(logr))
            swap = rbm1.theano_rng.binomial(n=1, p=r, size=(self.batch_size,), dtype=floatX)
            self.swap_funcs += [theano.function([], [swap, rbm1_negh])]
            self.swap_ratios += [0.]

    def do_swap(self, i, alpha=0.99):
        """ Perform swaps between samples of i-th and (i+1)-th RBM """
        assert i+1 < len(self.rbms)
        swap, rbm1_negh = self.swap_funcs[i]()
        rbm1_negv = self.rbms[i].neg_v.get_value()
        rbm2_negv = self.rbms[i+1].neg_v.get_value()

        rbm1_negv[swap == 1] = self.rbms[i].sample_v_given_h_func(rbm2_negv)[swap == 1]
        rbm2_negv[swap == 1] = rbm1_negh[swap == 1]
        self.rbms[i].neg_v.set_value(rbm1_negv)
        self.rbms[i+1].neg_v.set_value(rbm2_negv)
        self.swap_ratios[i] = alpha * self.swap_ratios[i] + (1.-alpha) * swap.mean()

    def do_swaps(self):
        for i in xrange(self.depth - 1):
            self.do_swap(i)

    def train_batch(self, dataset, batch_size):
        x = dataset.get_batch_design(batch_size, include_labels=False)

        t1 = time.time()
        # Compute approximate posterior at each layer
        xs = [x]
        for rbm in self.rbms[:-1]:
            xs += [rbm.post_func(xs[-1])]
        # Train each layer to model the previous layer's posterior
        for x, rbm in zip(xs, self.rbms):
            rbm.batch_train_func(x)
        self.do_swaps()
        self.cpu_time += time.time() - t1

        self.increase_timers()

        # save to different path each epoch
        r0 = self.rbms[0]
        if r0.my_save_path and \
           (self.batches_seen in r0.save_at or
            self.batches_seen % r0.save_every == 0):
            fname = r0.my_save_path + '_e%i.pkl' % self.batches_seen
            print 'Saving to %s ...' % fname,
            serial.save(fname, r0)
            print 'done'

        return self.batches_seen < self.max_updates

    def increase_timers(self):
        # accounting...
        self.examples_seen += self.batch_size
        self.batches_seen += 1
        for rbm in self.rbms:
            rbm.examples_seen += self.batch_size
            rbm.batches_seen += 1
            rbm.cpu_time = self.cpu_time

        if self.batches_seen % 100 == 0:
            print 'Swap ratios:', self.swap_ratios

    def monitor_matrix(self, w, name=None):
        if name is None: assert hasattr(w, 'name')
        name = name if name else w.name
        return {name + '.min':  w.min(axis=[0,1]),
                name + '.max':  w.max(axis=[0,1]),
                name + '.absmean': abs(w).mean(axis=[0,1])}

    def monitor_vector(self, b, name=None):
        if name is None: assert hasattr(b, 'name')
        name = name if name else b.name
        return {name + '.min':  b.min(),
                name + '.max':  b.max(),
                name + '.absmean': abs(b).mean()}

    def get_monitoring_channels(self, x, y=None):
        chans = OrderedDict()
        for i, rbm in enumerate(self.rbms):
            chans.update(self.monitor_matrix(rbm.Wv, name='Wv.%i'%i))
            normw = T.sqrt(T.sum(rbm.Wv**2, axis=0))
            chans.update(self.monitor_vector(normw, name='Wv_norm.%i'%i))
        return chans


class TrainingAlgorithm(default.DefaultTrainingAlgorithm):

    def setup(self, model, dataset):
        super(TrainingAlgorithm, self).setup(model, dataset)
