from pylearn2.training_callbacks.training_callback import TrainingCallback
from pylearn2.utils import serial

class pylearn2_save_callback(TrainingCallback):

    def __init__(self, my_save_path=None, save_at=None, save_every=None):
        self.my_save_path = my_save_path
        self.save_at = save_at
        self.save_every = save_every

    def __call__(self, model, train, algorithm):

        is_save_interval = (
                model.batches_seen in self.save_at or
                model.batches_seen % self.save_every == 0)

        if self.my_save_path and is_save_interval:
            fname = self.my_save_path + '_e%i.pkl' % model.batches_seen
            model.save_path = fname
            print 'Saving to %s ...' % fname,
            serial.save(fname, model)
            print 'done'


