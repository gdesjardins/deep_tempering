"""
This tutorial introduces restricted boltzmann machines (RBM) using Theano.

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections.
"""
import numpy
import md5
import pickle
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

def sigm(x): return 1./(1 + numpy.exp(-x))

class RBM(Model, Block):

    def validate_flags(self, flags):
        flags.setdefault('ml_vbias', 0)
        flags.setdefault('enable_centering', False)
        flags.setdefault('train_on_samples', False)
        flags.setdefault('sample_data', False)
        flags.setdefault('centered', True)
        if len(flags.keys()) != 5:
            raise NotImplementedError('One or more flags are currently not implemented.')

    @classmethod
    def quick_alloc(cls, n_v, n_h):
        lr_spec = {'type': 'linear', 'start':1e-3, 'end':1e-3}
        iscales={'Wv':0.01, 'vbias':0, 'hbias':0}
        sp_weight={'h':0.}
        sp_targ={'h':0.1}
        return cls(n_v = n_v, n_h = n_h,
                lr_spec = lr_spec,
                iscales = iscales,
                sp_weight = sp_weight,
                sp_targ = sp_targ)

    def __init__(self, 
            numpy_rng = None, theano_rng = None,
            n_h=99, n_v=100, init_from=None, neg_sample_steps=1,
            lr_spec=None, lr_mults = {},
            iscales={}, clip_min={}, clip_max={},
            l1 = {}, l2 = {},
            sp_weight={}, sp_targ={},
            batch_size = 13,
            compile=True, debug=False, seed=1241234,
            flags = {},
            max_updates = 5e5, **kwargs):
        """
        :param n_h: number of h-hidden units
        :param n_v: number of visible units
        :param iscales: optional dictionary containing initialization scale for each parameter
        :param neg_sample_steps: number of sampling updates to perform in negative phase.
        :param l1: hyper-parameter controlling amount of L1 regularization
        :param l2: hyper-parameter controlling amount of L2 regularization
        :param batch_size: size of positive and negative phase minibatch
        :param compile: compile sampling and learning functions
        :param seed: seed used to initialize numpy and theano RNGs.
        """
        Model.__init__(self)
        Block.__init__(self)
        assert lr_spec is not None
        for k in ['h']: assert k in sp_weight.keys()
        for k in ['h']: assert k in sp_targ.keys()
        self.validate_flags(flags)

        self.jobman_channel = None
        self.jobman_state = {}
        self.register_names_to_del(['jobman_channel'])

        ### make sure all parameters are floatX ###
        for (k,v) in l1.iteritems(): l1[k] = npy_floatX(v)
        for (k,v) in l2.iteritems(): l2[k] = npy_floatX(v)
        for (k,v) in sp_weight.iteritems(): sp_weight[k] = npy_floatX(v)
        for (k,v) in sp_targ.iteritems(): sp_targ[k] = npy_floatX(v)
        for (k,v) in clip_min.iteritems(): clip_min[k] = npy_floatX(v)
        for (k,v) in clip_max.iteritems(): clip_max[k] = npy_floatX(v)

        # dump initialization parameters to object
        for (k,v) in locals().iteritems():
            if k!='self': setattr(self,k,v)

        # allocate random number generators
        self.rng = numpy.random.RandomState(seed) if numpy_rng is None else numpy_rng
        self.theano_rng = RandomStreams(self.rng.randint(2**30)) if theano_rng is None else theano_rng

        ############### ALLOCATE PARAMETERS #################
        # allocate symbolic variable for input
        self.input = T.matrix('input')
        self.init_parameters()
        self.init_chains()

        # learning rate, with deferred 1./t annealing
        self.iter = sharedX(0.0, name='iter')

        if lr_spec['type'] == 'anneal':
            num = lr_spec['init'] * lr_spec['start'] 
            denum = T.maximum(lr_spec['start'], lr_spec['slope'] * self.iter)
            self.lr = T.maximum(lr_spec['floor'], num/denum) 
        elif lr_spec['type'] == '1_t':
            self.lr = npy_floatX(lr_spec['num']) / (self.iter + npy_floatX(lr_spec['denum']))
        elif lr_spec['type'] == 'delayed_1_t':
            _iter = self.iter - npy_floatX(lr_spec['delay'])
            self.lr = T.switch(
                    self.iter < lr_spec['delay'],
                    npy_floatX(lr_spec['init']),
                    npy_floatX(lr_spec['num']) / (_iter + npy_floatX(lr_spec['denum'])))
        elif lr_spec['type'] == 'linear':
            lr_start = npy_floatX(lr_spec['start'])
            lr_end   = npy_floatX(lr_spec['end'])
            self.lr = lr_start + self.iter * (lr_end - lr_start) / npy_floatX(self.max_updates)
        elif lr_spec['type'] == 'constant':
            self.lr = sharedX(lr_spec['value'], name='lr')
        else:
            raise ValueError('Incorrect value for lr_spec[type]')

        # configure input-space (new pylearn2 feature?)
        self.input_space = VectorSpace(n_v)
        self.output_space = VectorSpace(n_h)

        self.batches_seen = 0               # incremented on every batch
        self.examples_seen = 0              # incremented on every training example
        self.logz = sharedX(0.0, name='logz')
        self.cpu_time = 0

        self.error_record = []
 
        if compile: self.do_theano()

        if init_from:
            raise NotImplementedError()

    def init_weight(self, iscale, shape, name, normalize=False, axis=0):
        value =  self.rng.normal(size=shape) * iscale
        if normalize:
            value /= numpy.sqrt(numpy.sum(value**2, axis=axis))
        return sharedX(value, name=name)

    def init_parameters(self):
        # init weight matrices
        self.Wv = self.init_weight(self.iscales.get('Wv', 1.0), (self.n_v, self.n_h), 'Wv')
        # allocate shared variables for bias parameters
        self.vbias = sharedX(self.iscales['vbias'] * numpy.ones(self.n_v), name='vbias')
        self.hbias = sharedX(self.iscales['hbias'] * numpy.ones(self.n_h), name='hbias')
        self.cv = sharedX(numpy.zeros(self.n_v), name='cv')
        ch = numpy.ones(self.n_h) * (0.5 if self.flags['enable_centering'] else 0.)
        self.ch = sharedX(ch, name='ch')

    def init_parameters_from_data(self, x):
        if self.flags['ml_vbias']:
            self.vbias.set_value(rbm_utils.compute_ml_bias(x))
        if self.flags['enable_centering']:
            self.cv.set_value(x.mean(axis=0).astype(floatX))
 
    def init_chains(self):
        """ Allocate shared variable for persistent chain """
        self.neg_ev = sharedX(self.rng.rand(self.batch_size, self.n_v), name='neg_ev')
        self.neg_v  = sharedX(self.rng.rand(self.batch_size, self.n_v), name='neg_v')
        self.neg_h  = sharedX(self.rng.rand(self.batch_size, self.n_h), name='neg_h')
 
    def params(self):
        """
        Returns a list of learnt model parameters.
        """
        params = [self.Wv, self.vbias, self.hbias]
        return params

    def get_param_values(self):
        return [p.get_value() for p in self.params()]

    def do_theano(self):
        """ Compiles all theano functions needed to use the model"""

        init_names = dir(self)

        ###### All fields you don't want to get pickled (e.g., theano functions) should be created below this line
        # SAMPLING: NEGATIVE PHASE
        neg_updates = self.neg_sampling_updates(n_steps=self.neg_sample_steps, use_pcd=True)
        self.sample_func = theano.function([], [], updates=neg_updates)

        ##
        # HELPER FUNCTIONS
        ##
        self.fe_v_func = theano.function([self.input], self.free_energy_v(self.input))
        self.fe_h_func = theano.function([self.input], self.free_energy_h(self.input))
        self.post_func = theano.function([self.input], self.h_given_v(self.input))
        self.v_given_h_func = theano.function([self.input], self.v_given_h(self.input))
        self.h_given_v_func = theano.function([self.input], self.h_given_v(self.input))
        self.sample_v_given_h_func = theano.function([self.input], self.sample_v_given_h(self.input))
        self.sample_h_given_v_func = theano.function([self.input], self.sample_h_given_v(self.input))
        h = T.matrix('h')
        self.energy_func = theano.function([self.input, h], self.energy(self.input, h))

        ##
        # BUILD COST OBJECTS
        ##
        lcost = self.ml_cost(pos_v = self.input, neg_v = self.neg_v)
        spcost = self.get_sparsity_cost()
        regcost = self.get_reg_cost(self.l2, self.l1)

        ##
        # COMPUTE GRADIENTS WRT. COSTS
        ##
        main_cost = [lcost, spcost, regcost]
        learning_grads = costmod.compute_gradients(self.lr, self.lr_mults, *main_cost)

        ##
        # BUILD UPDATES DICTIONARY FROM GRADIENTS
        ##
        learning_updates = costmod.get_updates(learning_grads)
        learning_updates.update({
            self.iter: self.iter+1,
            self.logz: 0.0,
        })

        # build theano function to train on a single minibatch
        self.batch_train_func = function([self.input], [],
                                         updates=learning_updates,
                                         name='train_rbm_func')

        #######################
        # CONSTRAINT FUNCTION #
        #######################

        # enforce constraints function
        constraint_updates = self.get_constraint_updates()
        self.enforce_constraints = theano.function([],[], updates=constraint_updates)

        ###### All fields you don't want to get pickled should be created above this line
        final_names = dir(self)
        self.register_names_to_del( [ name for name in (final_names) if name not in init_names ])

        # Before we start learning, make sure constraints are enforced
        self.enforce_constraints()

    def get_constraint_updates(self):
        updates = OrderedDict() 
        ## clip parameters to maximum values (if applicable)
        for (k,v) in self.clip_max.iteritems():
            assert k in [param.name for param in self.params()]
            param = getattr(self, k)
            updates[param] = T.clip(param, param, v)
        ## clip parameters to minimum values (if applicable)
        for (k,v) in self.clip_min.iteritems():
            assert k in [param.name for param in self.params()]
            param = getattr(self, k)
            updates[param] = T.clip(updates.get(param, param), v, param)
        return updates
 
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
            x = (self.rng.random_sample(x.shape) < x).astype(floatX)

        t1 = time.time()
        self.sample_func()
        self.batch_train_func(x.astype(floatX))
        # invalidate partition function after update
        self.enforce_constraints()
        self.cpu_time += time.time() - t1

        # accounting...
        self.examples_seen += len(x)
        self.batches_seen += 1

        return self.batches_seen < self.max_updates

    def free_energy_v(self, v_sample):
        """
        Computes free-energy of visible samples.
        :param v_sample: T.matrix of shape (batch_size, n_v)
        """
        fe  = 0.
        fe -= T.dot(v_sample, self.vbias)
        fe += T.sum(T.dot(v_sample, self.Wv) * self.ch, axis=1)
        h_input = self.h_given_v_input(v_sample)
        fe -= T.sum(T.nnet.softplus(h_input), axis=1)
        return fe

    def free_energy_h(self, h_sample):
        """
        Computes free-energy of hidden samples.
        :param v_sample: T.matrix of shape (batch_size, n_v)
        """
        fe  = 0.
        fe -= T.dot(h_sample, self.hbias)
        fe += T.sum(T.dot(h_sample, self.Wv.T) * self.cv, axis=1)
        v_input = self.v_given_h_input(h_sample)
        fe -= T.sum(T.nnet.softplus(v_input), axis=1)
        return fe

    def energy(self, v_sample, h_sample):
        e = 0.
        e -= T.sum(numpy.dot(v_sample, self.Wv) * h_sample, axis=1)
        e -= T.dot(v_sample, self.vbias)
        e -= T.dot(h_sample, self.hbias)
        return e

    def __call__(self, v):
        return self.h_given_v(v)

    ######################################
    # MATH FOR CONDITIONAL DISTRIBUTIONS #
    ######################################

    def h_given_v_input(self, v_sample):
        return T.dot(v_sample - self.cv, self.Wv) + self.hbias

    def h_given_v(self, v_sample):
        h_mean = self.h_given_v_input(v_sample)
        return T.nnet.sigmoid(h_mean)

    def sample_h_given_v(self, v_sample, rng=None, size=None):
        """
        Generates sample from p(h | v)
        """
        h_mean = self.h_given_v(v_sample)
        rng = self.theano_rng if rng is None else rng
        h_sample = rng.binomial(size=(h_mean.shape[0], self.n_h),
                                n=1, p=h_mean, dtype=floatX)
        return h_sample

    def v_given_h_input(self, h_sample):
        return T.dot(h_sample - self.ch, self.Wv.T) + self.vbias

    def v_given_h(self, h_sample):
        """
        Computes the mean-activation of visible units, given all other variables.
        :param h_sample: T.matrix of shape (batch_size, n_h)
        """
        v_mean = self.v_given_h_input(h_sample)
        return T.nnet.sigmoid(v_mean)

    def sample_v_given_h(self, h_sample, rng=None, size=None):
        v_mean = self.v_given_h(h_sample)
        rng = self.theano_rng if rng is None else rng
        v_sample = rng.binomial(size=(v_mean.shape[0], self.n_v),
                                n=1, p=v_mean, dtype=floatX)
        return v_sample

    ##################
    # SAMPLING STUFF #
    ##################
    def neg_sampling(self, h_sample, v_sample, n_steps=1):
        """
        Gibbs step for negative phase, which alternates: p(h|v), p(v|h).
        :param h_sample: T.matrix of shape (batch_size, n_h)
        :param v_sample: T.matrix of shape (batch_size, n_v)
        :param n_steps: number of Gibbs updates to perform in negative phase.
        """
        def gibbs_iteration(h1, v1, size):
            h2  = self.sample_h_given_v(v1, size=size)
            v2  = self.sample_v_given_h(h2, size=size)
            ev2 = self.v_given_h(h2)
            return [h2, v2, ev2]

        [new_h, new_v, new_ev] , updates = theano.scan(
                gibbs_iteration,
                outputs_info = [h_sample, v_sample, None],
                non_sequences = [v_sample.shape[0]],
                n_steps=n_steps)

        return [new_h[-1], new_v[-1], new_ev[-1]]

    def neg_sampling_updates(self, n_steps=1, use_pcd=True):
        """
        Implements the negative phase, generating samples from p(h,s,v).
        :param n_steps: scalar, number of Gibbs steps to perform.
        """
        init_chain = self.neg_v if use_pcd else self.input
        [new_h, new_v, new_ev] =  self.neg_sampling(
                self.neg_h, self.neg_v, n_steps = n_steps)

        updates = OrderedDict()
        updates[self.neg_h] = new_h
        updates[self.neg_v] = new_v
        updates[self.neg_ev] = new_ev
        return updates

    def ml_cost(self, pos_v, neg_v):
        pos_cost = T.mean(self.free_energy_v(pos_v))
        neg_cost = T.mean(self.free_energy_v(neg_v))
        cost = pos_cost - neg_cost
        # build gradient of cost with respect to model parameters
        return costmod.Cost(cost, self.params(), [pos_v, neg_v])

    ##############################
    # GENERIC OPTIMIZATION STUFF #
    ##############################
    def get_sparsity_cost(self):
        hack_h = self.h_given_v(self.input)
        # define loss based on value of sp_type
        eps = npy_floatX(1e-5)
        loss = lambda targ, val: - npy_floatX(targ) * T.log(eps + val) \
                                 - npy_floatX(1-targ) * T.log(1 - val + eps)

        params = []
        cost = T.zeros((), dtype=floatX)
        if self.sp_weight['h']:
            params += [self.Wv, self.hbias]
            cost += self.sp_weight['h']  * T.sum(loss(self.sp_targ['h'], hack_h).mean(axis=0))

        return costmod.Cost(cost, params)

    def get_reg_cost(self, l2=None, l1=None):
        """
        Builds the symbolic expression corresponding to first-order gradient descent
        of the cost function ``cost'', with some amount of regularization defined by the other
        parameters.
        :param l2: dict whose values represent amount of L2 regularization to apply to
        parameter specified by key.
        :param l1: idem for l1.
        """
        cost = T.zeros((), dtype=floatX)
        params = []

        for p in self.params():

            if l1.get(p.name, 0):
                cost += l1[p.name] * T.sum(abs(p))
                params += [p]

            if l2.get(p.name, 0):
                cost += l2[p.name] * T.sum(p**2)
                params += [p]
            
        return costmod.Cost(cost, params)

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
        chans.update(self.monitor_matrix(self.Wv))
        chans.update(self.monitor_vector(self.vbias))
        chans.update(self.monitor_vector(self.hbias))
        chans.update(self.monitor_matrix(self.neg_h))
        chans.update(self.monitor_matrix(self.neg_v))
        chans.update(self.monitor_vector(self.cv))
        wv_norm = T.sqrt(T.sum(self.Wv**2, axis=0))
        chans.update(self.monitor_vector(wv_norm, name='wv_norm'))
        chans['lr'] = self.lr
        return chans

    def uncenter(self):
        assert self.flags['centered']
        cv = self.cv.get_value()
        ch = self.ch.get_value()
        self._backup = {'cv': cv, 'ch': ch}
        Wv = self.Wv.get_value()
        self.vbias.set_value(self.vbias.get_value() - numpy.dot(ch, Wv.T))
        self.hbias.set_value(self.hbias.get_value() - numpy.dot(cv, Wv))
        self.cv.set_value(numpy.zeros_like(cv))
        self.ch.set_value(numpy.zeros_like(ch))
        self.flags['centered'] = False

    def recenter(self):
        assert not self.flags['centered']
        Wv = self.Wv.get_value()
        self.vbias.set_value(self.vbias.get_value() + numpy.dot(self._backup['ch'], Wv.T))
        self.hbias.set_value(self.hbias.get_value() + numpy.dot(self._backup['cv'], Wv))
        self.cv.set_value(self._backup['cv'])
        self.ch.set_value(self._backup['ch'])
        del self._backup
        self.flags['centered'] = True


def reload_params(rbm, fname):

    fp = open(fname, 'r')
    model = pickle.load(fp)
    fp.close()
    rbm.Wv.set_value(model.Wv.get_value())
    rbm.hbias.set_value(model.hbias.get_value())
    rbm.vbias.set_value(model.vbias.get_value())
    rbm.neg_ev.set_value(model.neg_ev.get_value())
    rbm.neg_v.set_value(model.neg_v.get_value())
    rbm.neg_h.set_value(model.neg_h.get_value())

    # sync random number generators
    rbm.rng.set_state(model.rng.get_state())
    rbm.theano_rng.rstate = model.theano_rng.rstate
    for (rbm_rng_state, model_rng_state) in \
            zip(rbm.theano_rng.state_updates, 
                model.theano_rng.state_updates):
        rbm_rng_state[0].set_value(model_rng_state[0].get_value())

    # reset misc. attributes
    rbm.batches_seen = model.batches_seen
    rbm.examples_seen = model.examples_seen
    rbm.logz.set_value(model.logz.get_value())
    rbm.cpu_time = model.cpu_time
