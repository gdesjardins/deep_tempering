"""
IMPORTANT: actually runs on sampled MNIST, contrary to nips2013_mnist_tdbn_exp9_1,
which used thresholding to binarize (due to a discrepancy in Pylearn2).
"""

from jobman.tools import DD, flatten
from jobman import api0, sql
import numpy

from pylearn2.scripts.jobman import experiment

rng = numpy.random.RandomState(4312987)

if __name__ == '__main__':
    db = api0.open_db('postgres://opter.iro.umontreal.ca/desjagui_db/icml14_dtneg_mnist_rbm_exp1_1')

    state = DD()

    state.yaml_template = """
        !obj:pylearn2.train.Train {
            "dataset": &data !obj:pylearn2.datasets.mnist.MNIST {
                "which_set": 'train',
                "center": False,
                "one_hot": True,
                "binarize": 'sample',
            },
            "model": &model !obj:deep_tempering.rbm.RBM {
                "seed" : %(seed1)i,
                "batch_size" : &batch_size %(batch_size)i,
                "n_v"  : 784,
                "n_h"  : &nh1 %(nh1)i,
                "gibbs_vhv": True,
                "neg_sample_steps" : 1,
                "flags": {
                    'ml_vbias': 1,
                    'enable_centering': 1,
                },
                "lr_spec"  : {
                    'type': '1_t',
                    'num': %(lr_num)f,
                    'denum': %(lr_denum)f,
                },
                "l2": {
                    'Wv': %(rbm0_l2)f,
                },
                # WARNING: change default values before
                "lr_mults": {},
                "iscales" : {
                    'Wv': 0.01,
                    'hbias':0.,
                    'vbias':0.,
                },
                "sp_weight": {"h":0.},
                "sp_targ"  : {"h":0.2},
                "debug": True,
                "max_updates": 2000000,
            },
            "algorithm": !obj:deep_tempering.training_algorithm.TrainingAlgorithm {
                "batch_size": *batch_size,
                "batches_per_iter" : 1000,
                "monitoring_batches": 11,
                "monitoring_dataset": *data,
            },
            "extensions": [
                !obj:deep_tempering.scripts.likelihood.rbm_callback.pylearn2_rbm_likelihood_callback {
                    "interval": 100000,
                    "trainset": *data,
                    "layer": 0,
                },
                !obj:deep_tempering.scripts.likelihood.rbm_callback.pylearn2_rbm_likelihood_callback {
                    "interval": 100000,
                    "trainset": &testdata !obj:pylearn2.datasets.mnist.MNIST {
                        "which_set": 'test',
                        "center": False,
                        "one_hot": True,
                        "binarize": 'sample',
                    },
                    "layer": 0,
                },
                !obj:deep_tempering.save_callback.pylearn2_save_callback {
                    "save_every": 100000,
                    "save_at": [],
                    "my_save_path": "model",
                },
            ]
        }
    """

    #for seed in [123141, 71629, 92735]: #, 1230, 475629]:
    for seed in [123141]:

        for (lr_mult, lr_num, lr_denum) in [
                (1, 100, 20000),
                (1, 500, 100000),
                (1, 1000, 200000),
                (1, 2000, 400000),
                (1, 4000, 800000),
                (2, 2000, 400000)]:

            for batch_size in [10]:
                for nh1 in [500]:
                    for nh2 in [500]:
                        for rbm0_l2 in [0.,1e-1,1e-2,1e-3,1e-4]:
                            state.hyper_parameters = {
                                'seed1': seed,
                                'seed2': seed + 1,
                                'lr_mult': lr_mult,
                                'lr_num': lr_num,
                                'lr_denum': lr_denum,
                                'nh1': nh1,
                                'nh2': nh2,
                                'batch_size': batch_size,
                                'rbm0_l2': rbm0_l2,
                            }
                            
                            sql.insert_job(
                                    experiment.train_experiment,
                                    flatten(state),
                                    db,
                                    force_dup=True)
