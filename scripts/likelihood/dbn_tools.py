import numpy
import logging
import optparse
import time
import pickle
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from pylearn2.utils import serial
from pylearn2.datasets import mnist

floatX = theano.config.floatX
logging.basicConfig(level=logging.INFO)

def xent(p, k):
    return -k * numpy.log(p+1e-5) - (1. - k ) * numpy.log(1-p+1e-5)

def compute_likelihood_lbound(model, log_z, test_x):
    """
    Compute likelihood as below, where q is the variational approximation
    q(v,h1,h2,h3) := q(h1|v)q(h2|h1)q(h2,h3) to p(h1,h2,h3|v)

    Denote FE as the free-energy of RBM(h2,h3) after marginalizing out h3.

        ln p(v) \geq + E_[q] log p(v|h1) + E_[q] log p(h1|h2) + E_[q] [-FE(h2)] 
                     - E_[q] log p(h1|v) - E_[q] log p(h2|h1) - E_[q] log p(h3|h2)
                     - log Z_{23}

    See section 4.2 of AIS paper for details.

    Inputs:
    -------
    model: deep_tempering.tempered_dbn.TemperedDBN
    log_z: scalar
        Estimate partition function of last-layer RBM.
    test_x: numpy.ndarray
        Data to compute likelihood.

    Returns:
    --------
    Scalar, representing negative log-likelihood of test data under the model.
    """
    i = 0.
    eq_log_p = 0
    for i in xrange(0, len(test_x), model.batch_size):

        # recast data as floatX and apply preprocessing if required
        x = numpy.array(test_x[i:i + model.batch_size, :], dtype=floatX)
        eq_log_px = 0.

        # perform inference
        psamples = model.inference_func(x)

        # Main term from expected log p(h2,h3) under q.
        # psamples[-2] := h2
        eq_log_px += -model.rbms[-1].fe_v_func(psamples[-2])

        # Other likelihood terms of the form E_q [log p(h_i|h_{i+1})]
        # Notation: lhs := h_i, rhs:= h_{i+1}
        for rbm, lhs, rhs in zip(model.rbms, psamples[:-2], psamples[1:-1]):
            p_hi_given_hip1 = rbm.v_given_h_func(rhs)
            eq_log_px += -xent(p_hi_given_hip1, lhs).sum(axis=1)

        # Contribution of the entropy of h(q) to the variational lower bound
        hq = 0
        for psample in psamples[1:]:
            temp = xent(psample, psample)
            hq += numpy.sum(temp, axis=1)
        eq_log_px += hq

        # perform moving average of negative likelihood
        # divide by len(x) and not bufsize, since last buffer might be smaller
        eq_log_p = (i * eq_log_p + eq_log_px.sum()) / (i + len(x))

    eq_log_p -= log_z

    return eq_log_p
