import numpy
import pickle

from pylearn2.training_callbacks.training_callback import TrainingCallback
from deep_tempering.scripts.likelihood import rbm_tools
from deep_tempering.scripts.likelihood import dbn_tools
from deep_tempering.tempered_dbn import TemperedDBN
from deep_tempering.utils import logging

class pylearn2_dbn_exact_callback(TrainingCallback):

    def __init__(self, trainset, interval=10):
        self.trainset = trainset
        self.interval = interval
        self.logger = logging.HDF5Logger('dbn_callback.hdf5')

        self.jobman_results = {
                'best_batches_seen': 0,
                'best_cpu_time': 0,
                'best_train_ll': -numpy.Inf,
                'best_logz': 0.,
                }

    def __call__(self, model, train, algorithm):
        if model.batches_seen == 0:
            return
        if (model.batches_seen % self.interval) != 0:
            return

        rbm = model.rbms[-1]
        logz = rbm_tools.compute_log_z(rbm, rbm.fe_v_func)
        train_ll = dbn_tools.compute_likelihood_lbound(model, logz, self.trainset.X)

        self.log(model, train_ll, logz)
        if model.jobman_channel:
            model.jobman_channel.save()

    def log(self, model, train_ll, logz):

        # log to database
        self.jobman_results['batches_seen'] = model.batches_seen
        self.jobman_results['cpu_time'] = model.cpu_time
        self.jobman_results['train_ll'] = train_ll
        self.jobman_results['logz'] = logz
        if train_ll > self.jobman_results['best_train_ll']:
            self.jobman_results['best_batches_seen'] = self.jobman_results['batches_seen']
            self.jobman_results['best_cpu_time'] = self.jobman_results['cpu_time']
            self.jobman_results['best_train_ll'] = self.jobman_results['train_ll']
            self.jobman_results['best_logz'] = self.jobman_results['logz']
        model.jobman_state.update(self.jobman_results)

        self.logger.log_list(model.batches_seen,
                [('train_ll', '%.3f', train_ll),
                 ('logz', '%.3f', logz)])
