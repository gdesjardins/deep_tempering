import numpy
from pylearn2.training_algorithms import default

class TrainingAlgorithm(default.DefaultTrainingAlgorithm):

    def setup(self, model, dataset):

        # Initialize some model parameters based on training data.
        if hasattr(dataset, 'X'):
            x = dataset.X
        elif i == 0:
            x = dataset.get_batch_design(1000, include_labels=False)
        model.init_parameters_from_data(x)

        dataset._iterator = dataset.iterator(
                mode='shuffled_sequential',
                batch_size=model.batch_size)


        super(TrainingAlgorithm, self).setup(model, dataset)
