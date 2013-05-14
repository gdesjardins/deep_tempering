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

def bernouilli(rng, mean):
    return (rng.random_sample(mean.shape) < mean).astype(floatX)

def compute_likelihood_lbound(model, log_z, test_x, M=5):
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
    M: int
        See Eq.(23) of AIS paper for details.

    Returns:
    --------
    Scalar, representing negative log-likelihood of test data under the model.
    """

    i = 0.
    eq_log_p = 0
    for i in xrange(0, len(test_x), model.batch_size):

        # recast data as floatX and apply preprocessing if required
        x = numpy.array(test_x[i:i + model.batch_size, :], dtype=floatX)

        # compute mean-field approximation to posterior
        posmfs = model.inference_func(x)

        # ln p(v,h_1, ..., h_{L-1}, h_L) = ln p(v|h1) +... ln p(h_{L-2}|h_{L-1}) + ln p(h_{L-1})
        # where we have marginalized out h_L in the last term of the summation.
        _eq_log_px = 0

        # Estimate E_q[ln p(h_{L-1})]
        for j in xrange(M):

            # Generate samples from mean-field distributions
            samples = [posmfs[0]]
            for i in range(1,len(posmfs)):
                samples += [bernouilli(model.rbms[i-1].rng, posmfs[i])]

            _eq_log_px += -model.rbms[-1].fe_v_func(samples[-2])

            # Estimate E_q[ ln p(h_i | h_{i+1}) ], for all i < L-1
            for rbm, hi, hip1 in zip(model.rbms[:-1], samples[:-2], samples[1:-1]):
                p_hi_given_hip1 = rbm.v_given_h_func(hip1)
                _eq_log_px += -xent(p_hi_given_hip1, hi).sum(axis=1)
            
        eq_log_px = _eq_log_px / M

        # Contribution of the entropy of h(q) to the variational lower bound.
        # Do not count entropy of p(h_L | h_{L-1}), since h_L has been marginalized.
        hq = 0
        for posmf in posmfs[1:-1]:
            hq += xent(posmf, posmf).sum(axis=1)
        eq_log_px += hq

        # perform moving average of negative likelihood
        # divide by len(x) and not bufsize, since last buffer might be smaller
        eq_log_p = (i * eq_log_p + eq_log_px.sum()) / (i + len(x))

    eq_log_p -= log_z

    return eq_log_p

def logmean(x, axis=0):
    return numpy.log(numpy.exp(x - x.max()).mean(axis=axis)) + x.max()

def compute_likelihood_lbound_theis(model, log_z, test_x, M=5):
    """
    See Figure 4 of "In all likelihood, deep belief is not enough", Theis et al.
    """

    i = 0.
    ulogprob = 0

    for i in xrange(0, len(test_x), model.batch_size):

        # recast data as floatX and apply preprocessing if required
        x = numpy.array(test_x[i:i + model.batch_size, :], dtype=floatX)

        logiws = numpy.zeros((M, model.batch_size)) 

        for j in xrange(M):
            sample = x
            for rbm in model.rbms[:-1]:
                logiws[j, :] += -rbm.fe_v_func(sample)
                sample = rbm.sample_h_given_v_func(sample)
                logiws[j, :] += -rbm.fe_h_func(sample)
            logiws[j, :] += model.rbms[-1].fe_v_func(sample)

        ulogprob_x = logmean(logiws, axis=0)

        # perform moving average of negative likelihood
        # divide by len(x) and not bufsize, since last buffer might be smaller
        ulogprob = (i * ulogprob + ulogprob_x.sum()) / (i + len(x))

    logprob = ulogprob - log_z

    return logprob
