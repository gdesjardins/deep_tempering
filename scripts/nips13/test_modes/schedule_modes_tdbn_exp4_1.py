"""
2-layer TDBN architecture. I noticed that swap rates at lower learning rates drop very quickly
to 0. This experiment tests the effect of higher learning rates in the upper layers. If mixing
is better at the upper-layers, higher learning rates should be possible.
"""

from jobman.tools import DD, flatten
from jobman import api0, sql
import numpy

from pylearn2.scripts.jobman import experiment

rng = numpy.random.RandomState(4312987)

if __name__ == '__main__':
    db = api0.open_db('postgres://opter.iro.umontreal.ca/desjagui_db/nips2013_modes_tdbn_exp4_1')

    state = DD()

    state.yaml_template = """
        !obj:pylearn2.scripts.train.Train {
            "dataset": &data !obj:deep_tempering.data.test_modes.OnlineModesNIPSDLUFL {},
            "model": &model !obj:deep_tempering.tempered_dbn.TemperedDBN {
                "rbm1": &rbm1 !obj:deep_tempering.rbm.RBM {
                        "seed" : %(seed1)i,
                        "batch_size" : &batch_size %(batch_size)i,
                        "n_v"  : 784,
                        "n_h"  : &nh1 %(nh1)i,
                        "gibbs_vhv": True,
                        "neg_sample_steps" : 1,
                        "flags": {
                            'ml_vbias': 0,
                        },
                        "lr_spec"  : {
                            'type': 'linear',
                            'start': %(lr_start1)f,
                            'end': %(lr_end1)f,
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
                        "save_every": 1000000,
                        "save_at": [],
                        "max_updates": 1000000,
                        "my_save_path": "model",
                },
                "rbm2": &rbm2 !obj:deep_tempering.rbm.RBM {
                        "seed" : %(seed2)i,
                        "batch_size" : *batch_size,
                        "n_v"  : *nh1,
                        "n_h"  : &nh2 %(nh2)i,
                        "gibbs_vhv": True,
                        "neg_sample_steps" : 1,
                        "flags": {
                            'ml_vbias': 0,
                        },
                        "lr_spec"  : {
                            'type': 'linear',
                            'start': %(lr_start2)f,
                            'end': %(lr_end2)f,
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
                        "save_every": 1000000,
                        "save_at": [],
                        "max_updates": 1000000,
                        "my_save_path": "model",
                },
                "uni": &uni !obj:deep_tempering.rbm.RBM {
                        "seed" : %(seed2)i,
                        "batch_size" : *batch_size,
                        "n_v"  : *nh2,
                        "n_h"  : 1,
                        "gibbs_vhv": True,
                        "neg_sample_steps" : 1,
                        "flags": {
                            'ml_vbias': 0,
                        },
                        "lr_spec"  : {
                            'type': 'linear',
                            'start': %(lr_start3)f,
                            'end': %(lr_end3)f,
                        },
                        # WARNING: change default values before
                        "lr_mults": {
                            'Wv': 0.,
                            'hbias': 0.,
                            'vbias': 1.,
                        },
                        "iscales" : {
                            'Wv': 0.,
                            'hbias':0.,
                            'vbias':0.,
                        },
                        "sp_weight": {"h":0.},
                        "sp_targ"  : {"h":0.2},
                        "debug": True,
                        "save_every": 1000000,
                        "save_at": [],
                        "max_updates": 1000000,
                        "my_save_path": "model",
                },
                "rbms": [*rbm1, *rbm2, *uni],
            },
            "algorithm": !obj:deep_tempering.tempered_dbn.TrainingAlgorithm {
                   "batch_size": *batch_size,
                   "batches_per_iter" : 1000,
                   "monitoring_batches": 11,
                   # not used but required by pylearn2
                   "monitoring_dataset": !obj:pylearn2.datasets.mnist.MNIST {
                        "which_set": 'train',
                        "one_hot": 1.,
                    },
            },
            "callbacks": [
                !obj:deep_tempering.scripts.likelihood.exactll_callback.pylearn2_exactll_callback {
                    "interval": 1000,
                    "trainset": *data,
                },
                !obj:deep_tempering.scripts.likelihood.log_swap_callback.pylearn2_log_swap_callback {
                    "interval": 1000,
                },
            ]
        }
    """

    batch_size = 5
    nh1 = 10
    nh2 = 10
    for seed in [123141, 71629, 92735, 1230, 34789]:
        for lr1 in [1e-3, 1e-4, 1e-5]:
            for lr2_mult in [1., 10., 100.]:

                lr2 = lr1 * lr2_mult

                state.hyper_parameters = {
                    'seed1': seed,
                    'seed2': seed + 1,
                    'lr_start1': lr1,
                    'lr_end1': lr1,
                    'lr_start2': lr2,
                    'lr_end2': lr2,
                    'lr_start3': lr2,
                    'lr_end3': lr2,
                    'nh1': nh1,
                    'nh2': nh2,
                    'batch_size': batch_size,
                }
                
                sql.insert_job(
                        experiment.train_experiment,
                        flatten(state),
                        db,
                        force_dup=True)
