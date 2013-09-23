import theano
import numpy
import pickle

from pylearn2.train_extensions import TrainExtension
from deep_tempering.scripts.likelihood import rbm_tools
from deep_tempering.scripts.likelihood import dbn_tools
from deep_tempering.tempered_dbn import TemperedDBN
from deep_tempering.utils import logging
floatX = theano.config.floatX
npy_floatX = getattr(numpy, floatX)

class pylearn2_dbn_likelihood_callback(TrainExtension):

    def __init__(self, trainset, interval=10, layer=-1):
        """
        :param trainset: pylearn2 dataset object.
        :param interval: run callback every `this` number of iterations
        """
        self.trainset = trainset
        self.interval = interval
        self.layer = layer
        fname = 'dbn%i_%s_callback.hdf5' % (layer, trainset.args['which_set'])
        self.logger = logging.HDF5Logger(fname)

        self.jobman_results = {
                'best_batches_seen': 0,
                'best_cpu_time': 0,
                'best_train_ll': -numpy.Inf,
                'best_logz': 0.,
                'best_var_logz': 0.,
                }

    def on_monitor(self, model, dataset, algorithm):
        assert self.layer < len(model.rbms)
        if model.batches_seen == 0:
            return
        if (model.batches_seen % self.interval) != 0:
            return

        model.uncenter()

        rbm = model.rbms[self.layer]

        logz = rbm.logz.get_value()
        if not logz:
            if (rbm.n_h <= 25):
                (logz, var_logz) = rbm_tools.compute_log_z(rbm, rbm.fe_h_func), 0.
            elif (rbm.n_v <= 25):
                (logz, var_logz) = rbm_tools.compute_log_z(rbm, rbm.fe_v_func), 0.
            else:
                (logz, var_logz), _aisobj = rbm_tools.rbm_ais(
                        rbm.get_param_values(),
                        n_runs=100)
            rbm.logz.set_value(npy_floatX(logz))
            rbm.var_logz = var_logz

        train_ll = dbn_tools.compute_likelihood_lbound(model, logz, self.trainset.X)

        # recenter model
        model.recenter()

        self.log(model, train_ll, logz, rbm.var_logz)
        if model.jobman_channel:
            model.jobman_channel.save()

    def log(self, model, train_ll, logz, var_logz):

        # log to database
        self.jobman_results['batches_seen'] = model.batches_seen
        self.jobman_results['cpu_time'] = model.cpu_time
        self.jobman_results['train_ll'] = train_ll
        self.jobman_results['logz'] = float(logz)
        self.jobman_results['var_logz'] = float(var_logz)
        if train_ll > self.jobman_results['best_train_ll']:
            self.jobman_results['best_batches_seen'] = self.jobman_results['batches_seen']
            self.jobman_results['best_cpu_time'] = self.jobman_results['cpu_time']
            self.jobman_results['best_train_ll'] = self.jobman_results['train_ll']
            self.jobman_results['best_logz'] = self.jobman_results['logz']
            self.jobman_results['best_var_logz'] = self.jobman_results['var_logz']
        model.jobman_state.update(self.jobman_results)

        self.logger.log_list(model.batches_seen,
                [('train_ll', '%.3f', train_ll),
                 ('logz', '%.3f', logz),
                 ('var_logz', '%.3f', var_logz)])
