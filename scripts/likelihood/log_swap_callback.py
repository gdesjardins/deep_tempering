import numpy
import pickle

from pylearn2.training_callbacks.training_callback import TrainingCallback
from deep_tempering.scripts.likelihood import rbm_tools
from deep_tempering.tempered_dbn import TemperedDBN
from deep_tempering.utils import logging

class pylearn2_log_swap_callback(TrainingCallback):

    def __init__(self, interval=10):
        self.interval = interval
        self.logger = logging.HDF5Logger('swap_callback.hdf5')

    def __call__(self, model, train, algorithm):
        if (model.batches_seen % self.interval) != 0:
            return

        data = []
        for i, sr in enumerate(model.swap_ratios):
            data += [('swap%i' % i, '%.3f', sr)]
        self.logger.log_list(model.batches_seen, data)
