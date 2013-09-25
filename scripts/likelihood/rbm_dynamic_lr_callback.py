import theano
import numpy
import pickle

from pylearn2.train_extensions import TrainingExtension
from deep_tempering.scripts.likelihood import rbm_tools
from deep_tempering.tempered_dbn import TemperedDBN
from deep_tempering.rbm import reload_params
from deep_tempering.utils import logging
floatX = theano.config.floatX
npy_floatX = getattr(numpy, floatX)

class pylearn2_dynamic_lr_callback(TrainingExtension):
    """
    Adjusts the learning rates for all layers of the DBN, based on the estimated
    training set likelihood (of the data under the first layer model). When we
    detect a decrease in likelihood, the learning rate for all RBMs is halved.
    """

    def __init__(self, trainset, interval=10, layer=0):
        assert layer == 0
        self.trainset = trainset
        self.interval = interval
        self.layer = layer
        fname = 'rbm%i_%s_callback.hdf5' % (layer, trainset.args['which_set'])
        self.logger = logging.HDF5Logger(fname)

        self.jobman_results = {
                'best_batches_seen': 0,
                'best_cpu_time': 0,
                'best_train_ll': -numpy.Inf,
                'best_logz': 0.,
                'best_var_logz': 0.,
                }

    def on_monitor(self, model, dataset, algorithm):
        assert model.flags['pretrain'] == False
        for rbm in model.rbms:
            assert rbm.lr_spec['type'] == 'constant'

        if model.batches_seen == 0:
            return
        if (model.batches_seen % self.interval) != 0:
            return

        rbm = model.rbms[self.layer] if isinstance(model, TemperedDBN) else model
        train_ll = self.estimate_likelihood(model, rbm)
        print 'rbm%i: current=%f\t best=%f' % (self.layer, train_ll, self.jobman_results['best_train_ll'])

        is_increasing = train_ll >= self.jobman_results['best_train_ll']

        if is_increasing:
            self.log(rbm, train_ll, rbm.logz.get_value(), rbm.var_logz)
        else:
            self.decrease_lr(model)

        if model.jobman_channel:
            rbm.jobman_state.update(self.jobman_results)
            model.jobman_channel.save()

    def preproc(self, model, x):
        """
        Helper function which generates the representation at layer `self.layer`.
        """
        if isinstance(model, TemperedDBN):
            for rbm in model.rbms[:self.layer]:
                x = rbm.post_func(x)
        return x

    def estimate_likelihood(self, model, rbm):
        logz = rbm.logz.get_value()
        if not logz:
            # Estimate partition function
            if (rbm.n_h <= 25):
                (logz, var_logz) = rbm_tools.compute_log_z(rbm, rbm.fe_h_func), 0.
            elif (rbm.n_v <= 25):
                (logz, var_logz) = rbm_tools.compute_log_z(rbm, rbm.fe_v_func), 0.
            else:
                (logz, var_logz), _aisobj = rbm_tools.rbm_ais(
                        rbm.get_uncentered_param_values(),
                        n_runs=100,
                        data = self.trainset.X,
                        preproc = lambda x: self.preproc(model, x))
            rbm.logz.set_value(npy_floatX(logz))
            rbm.var_logz = var_logz

        train_ll = rbm_tools.compute_likelihood(rbm,
                data = self.trainset.X,
                log_z = logz,
                free_energy_fn = rbm.fe_v_func,
                preproc = lambda x: self.preproc(model, x))

        return train_ll
 
    def log(self, rbm, train_ll, logz, var_logz):

        # log to database
        self.jobman_results['batches_seen'] = rbm.batches_seen
        self.jobman_results['cpu_time'] = rbm.cpu_time
        self.jobman_results['train_ll'] = train_ll
        self.jobman_results['logz'] = float(logz)
        self.jobman_results['var_logz'] = float(var_logz)
        self.jobman_results['best_batches_seen'] = self.jobman_results['batches_seen']
        self.jobman_results['best_cpu_time'] = self.jobman_results['cpu_time']
        self.jobman_results['best_train_ll'] = self.jobman_results['train_ll']
        self.jobman_results['best_logz'] = self.jobman_results['logz']
        self.jobman_results['best_var_logz'] = self.jobman_results['var_logz']

        # log to hdf5
        self.logger.log_list(rbm.batches_seen,
                [('train_ll', '%.3f', train_ll),
                 ('logz', '%.3f', logz),
                 ('var_logz', '%.3f', var_logz),
                 ('lr', '%.6f', rbm.lr.get_value())])

    def decrease_lr(self, model):
        print '*** Train likelihood decreased. Halving learning rate. ***'
        for i, rbm in enumerate(model.rbms):
            assert rbm.flags['learn']
            print 'Reloading rbm%i from previous checkpoint...' % i,
            reload_params(rbm, rbm.fname)
            print 'done.'
            #rbm.lr.set_value(npy_floatX(0.1 * rbm.lr.get_value()))
            rbm.lr.set_value(0.0)
