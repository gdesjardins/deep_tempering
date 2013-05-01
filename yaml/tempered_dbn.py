!obj:pylearn2.scripts.train.Train {
    "dataset": &data !obj:pylearn2.datasets.mnist.MNIST {
        "which_set": 'train',
        "one_hot": 1.,
    },
    "model": &model !obj:deep_tempering.tempered_dbn.TemperedDBN {
        "rbm1": &rbm1 !obj:deep_tempering.rbm.RBM {
                "seed" : 123141,
                "batch_size" : &batch_size 100,
                "n_v"  : 784,
                "n_h"  : &nh1 500,
                "gibbs_vhv": True,
                "neg_sample_steps" : 1,
                "flags": {},
                "lr_spec"  : {
                    'type': 'linear',
                    'start': 0.010000,
                    'end': 0.010000
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
                "n_h"  : &nh2 2000,
                "gibbs_vhv": True,
                "neg_sample_steps" : 1,
                "flags": {},
                "lr_spec"  : {
                    'type': 'linear',
                    'start': 0.10000,
                    'end': 0.10000
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
        "rbm3": &rbm3 !obj:deep_tempering.rbm.RBM {
                "seed" : 123141,
                "batch_size" : *batch_size,
                "n_v"  : *nh2,
                "n_h"  : &nh3 2000,
                "gibbs_vhv": True,
                "neg_sample_steps" : 1,
                "flags": {},
                "lr_spec"  : {
                    'type': 'linear',
                    'start': 0.0010000,
                    'end': 0.0010000
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
        "rbms": [*rbm1, *rbm2], #, *rbm3],
    },
    "algorithm": !obj:deep_tempering.tempered_dbn.TrainingAlgorithm {
               "batch_size": *batch_size,
               "batches_per_iter" : 1000,
               "monitoring_batches": 11,
               "monitoring_dataset": *data,
    },
    "callbacks": [],
}

