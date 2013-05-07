from jobman.tools import DD, flatten
from jobman import api0, sql
import numpy

from pylearn2.scripts.jobman import experiment

rng = numpy.random.RandomState(4312987)

if __name__ == '__main__':
    db = api0.open_db('postgres://opter.iro.umontreal.ca/desjagui_db/nips2013_modes_tdbn_exp1_1')

    state = DD()

    state.yaml_template = """
        !obj:pylearn2.scripts.train.Train {
            "dataset": &data !obj:deep_tempering.data.test_modes.OnlineModesNIPSDLUFL {},
            "model": &model !obj:deep_tempering.tempered_dbn.TemperedDBN {
                "rbm1": &rbm1 !obj:deep_tempering.rbm.RBM {
                        "seed" : %(seed)i,
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
                            'start': %(lr_start)f,
                            'end': %(lr_end)f,
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
                        "save_every": 10000,
                        "save_at": [],
                        "max_updates": 10000000,
                        "my_save_path": "model",
                },
                "rbm2": &rbm2 !obj:deep_tempering.rbm.RBM {
                        "seed" : 123141,
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
                            'start': %(lr_start)f,
                            'end': %(lr_end)f,
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
                        "save_every": 10000,
                        "save_at": [],
                        "max_updates": 10000000,
                        "my_save_path": "model",
                },
                "rbms": [*rbm1, *rbm2],
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
                }
            ]
        }
    """

    for seed in [123141, 71629, 92735, 1230, 34789]:
        for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
            for batch_size in [5]:
                for nh1 in [10]:
                    for nh2 in [10,20]:
                        state.hyper_parameters = {
                            'seed': seed,
                            'lr_start': lr,
                            'lr_end': lr,
                            'nh1': nh1,
                            'nh2': nh2,
                            'batch_size': batch_size,
                        }
                        
                        sql.insert_job(
                                experiment.train_experiment,
                                flatten(state),
                                db,
                                force_dup=True)
