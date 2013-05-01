import numpy
import pickle

from pylearn2.training_callbacks.training_callback import TrainingCallback
from deep_tempering.scripts.likelihood import rbm_tools

class pylearn2_ais_callback(TrainingCallback):

    def __init__(self, trainset, ais_interval=10):

        self.trainset = trainset
        self.ais_interval = ais_interval

        self.pkl_results = {
                'batches_seen': [],
                'cpu_time': [],
                'train_ll': [],
                'logz': [],
                }

        self.jobman_results = {
                'best_batches_seen': 0,
                'best_cpu_time': 0,
                'best_train_ll': -numpy.Inf,
                'best_logz': 0,
                }
        fp = open('ais_callback.log','w')
        fp.write('Epoch\tBatches\tCPU\tTrain\tTest\tlogz\n')
        fp.close()

    def __call__(self, model, train, algorithm):
        if model.batches_seen == 0:
            return
        if (model.batches_seen % self.ais_interval) != 0:
            return

        (logz, var_logz) = rbm_tools.rbm_ais(
                [model.Wv, model.vbias, model.hbias],
                n_runs=100,
                data = self.trainset.X)

        train_ll = rbm_tools.compute_nll(model, self.trainset.X, logz,
                model.free_energy_v)

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

        # save to text file
        fp = open('ais_callback.log','a')
        fp.write('%i\t%f\t%f\t%f\n' % (
            self.jobman_results['batches_seen'],
            self.jobman_results['cpu_time'],
            self.jobman_results['train_ll'],
            self.jobman_results['logz']))
        fp.close()

        # save to pickle file
        self.pkl_results['batches_seen'] += [model.batches_seen]
        self.pkl_results['cpu_time'] += [model.cpu_time]
        self.pkl_results['train_ll'] += [train_ll]
        self.pkl_results['logz'] += [logz]
        fp = open('ais_callback.pkl','w')
        pickle.dump(self.pkl_results, fp)
        fp.close()


