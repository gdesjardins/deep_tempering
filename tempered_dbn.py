import numpy
import copy
import time
import pickle
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

    def validate_flags(self, flags):
        flags.setdefault('train_on_samples', False)
        flags.setdefault('sample_data', False)
        flags.setdefault('pretrain', False)
        flags.setdefault('order', 'swap_sample_learn')
        if len(flags.keys()) != 4:
            raise notimplementederror('one or more flags are currently not implemented.')

    def __init__(self, rbms=None, max_updates=1e6, flags={}):
        Model.__init__(self)
        Block.__init__(self)
        self.jobman_channel = None
        self.jobman_state = {}
        self.validate_flags(flags)
        self.register_names_to_del(['jobman_channel'])

        # dump initialization parameters to object
        for (k,v) in locals().iteritems():
            if k!='self': setattr(self,k,v)

        # validate that RBMs have the same number of units.
        for (rbm1, rbm2) in zip(rbms[:-1], rbms[1:]):
            assert rbm1.n_h == rbm2.n_v
            assert rbm1.batch_size == rbm2.batch_size
            #assert rbm1.flags['enable_centering']
            #assert rbm2.flags['enable_centering']
        self.rbms = rbms
        self.depth = len(rbms)
        self.rng = self.rbms[0].rng

        # configure input-space (necessary evil)
        self.input_space = VectorSpace(self.rbms[0].n_v)
        self.output_space = VectorSpace(self.rbms[-1].n_h)

        self.batches_seen = 0  # incremented on every batch
        self.examples_seen = 0 # incremented on every training example
        self.batch_size = self.rbms[0].batch_size
        self.cpu_time = 0
        self.init_train_sequence()
        self.do_theano()

    def init_parameters_from_data(self, x):
        """
        Override default model initialization, using training data statistics.
        """
        for i, rbm in enumerate(self.rbms):
            if i == 0:
                rbm.init_parameters_from_data(x)
            else:
                new_x = numpy.ones((1, rbm.n_v)) * 0.5
                rbm.init_parameters_from_data(new_x)

    def uncenter(self):
        for rbm in self.rbms:
            rbm.uncenter()

    def recenter(self):
        for rbm in self.rbms:
            rbm.recenter()

    def do_theano(self):
        init_names = dir(self)
        ###### All fields you don't want to get pickled should be created below this line
        self.build_swap_funcs()
        self.build_inference_func(sample=True)
        self.build_inference_func(sample=False)
        ###### All fields you don't want to get pickled should be created above this line
        final_names = dir(self)
        self.register_names_to_del( [ name for name in (final_names) if name not in init_names ])

    def init_train_sequence(self):
        self.rbms[0].flags['learn'] = True
        for rbm in self.rbms[1:]:
            rbm.flags['learn'] = False if self.flags['pretrain'] else True

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

    def build_inference_func(self, sample=False):
        rval = []
        layer_in = self.rbms[0].input
        for rbm in self.rbms:
            if sample:
                layer_out = rbm.sample_h_given_v(layer_in)
            else:
                layer_out = rbm.h_given_v(layer_in)
            rval += [layer_in]
            layer_in = layer_out
        rval += [layer_in]

        func = theano.function([self.rbms[0].input], rval)
        if sample:
            self.inference_sample_func = func
        else:
            self.inference_func = func

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
            if len(self.rbms) == 2:
                # always swap for 2-layer model
                self.do_swap(i)
            else:
                # When using > 2 layers, we swap RBMs (i,i+1) with even i, on
                # even iterations, and RBMs (i,i+1) with odd i, on odd
                # iterations.
                if i % 2 == 0 and self.batches_seen % 2 == 0:
                    # swap even layers at even iterations
                    self.do_swap(i)
                elif i % 2 == 1 and self.batches_seen % 2 == 1:
                    # swap odd layers at odd iterations
                    self.do_swap(i)

    def do_sample(self):
        for rbm in self.rbms:
            rbm.sample_func()

    def do_learn(self, x):
        for rbm in self.rbms:
            if rbm.flags['learn']:
                rbm.batch_train_func(x)
            if self.flags['train_on_samples']:
                x = rbm.sample_h_given_v_func(x)
            else:
                x = rbm.h_given_v_func(x)

    def train_batch(self, dataset, batch_size):

        try:
            x = dataset._iterator.next()
        except StopIteration:
            if hasattr(dataset._iterator._subset_iterator, 'shuffe'):
                dataset._iterator._subset_iterator.shuffle()
            else:
                dataset._iterator._subset_iterator.reset()
            x = dataset._iterator.next()

        if self.flags['sample_data']:
            x = self.rng.random_sample(x.shape) < x

        t1 = time.time()
        if self.flags['order'] == 'sample_swap_learn':
            self.do_sample()
            if not self.flags['pretrain']:
                self.do_swaps()
            self.do_learn(x.astype(floatX))
        elif self.flags['order'] == 'swap_sample_learn':
            if not self.flags['pretrain']:
                self.do_swaps()
            self.do_sample()
            self.do_learn(x.astype(floatX))
        else:
            raise ValueError('Invalid setting for flags[order]')

        self.cpu_time += time.time() - t1
        self.increase_timers()

        return self.batches_seen < self.max_updates

    def increase_timers(self):
        """ Synchronize various timers across all RBMs under simulation """
        self.examples_seen += self.batch_size
        self.batches_seen += 1
        for rbm in self.rbms:
            rbm.examples_seen += self.batch_size
            rbm.batches_seen += 1
            rbm.cpu_time = self.cpu_time

        if self.batches_seen % 100 == 0:
            print 'Swap ratios:', self.swap_ratios

    def get_monitoring_channels(self, x, y=None):
        chans = OrderedDict()
        for i, rbm in enumerate(self.rbms):
            ichans = rbm.get_monitoring_channels(x, y=y)
            for (k,v) in ichans.iteritems():
                chans['%s.%i' % (k,i)] = v
        return chans

    def __getstate__(self):
        # save each RBM in a separate pickle file
        self.rbm_fnames = []
        for i, rbm in enumerate(self.rbms):
            fname = '.rbm%i_e%i.pkl' % (i, self.batches_seen)
            # Always log filename of most recently saved model
            # TODO: move functionality to serial.save
            rbm.fname = fname
            serial.save(fname, rbm)
            self.rbm_fnames.append(fname)

        rval = super(TemperedDBN, self).__getstate__()
        rval.pop('rbms')
        return rval

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.rbms = []
        for i, fname in enumerate(self.rbm_fnames):
            fp = open(fname)
            self.rbms += [serial.load(fname)]
            fp.close()
        d.pop('rbm_fnames')
