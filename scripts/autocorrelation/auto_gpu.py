import numpy
_average_face = "/data/lisa/exp/luoheng/TFD/TFD_loca_contrast_normalization_mean_face.npy"

BASE = "/data/lisatmp/desjagui/expdir/deep_tempering/briaree/nips13_deeptempering/desjagui_db/"

import theano
import theano.tensor as T

v0   = T.tensor4('v0', dtype='float32')
vi   = T.tensor4('vtau', dtype='float32')
mean = T.TensorType('float32', [True,False,False])()

hat_v0 = v0 - mean
hat_vi = vi - mean
v0_sigma = T.sum(hat_v0**2, axis=[0,2,3])
vi_sigma = T.sum(hat_vi**2, axis=[0,2,3])

ri = T.sum(hat_v0 * hat_vi, axis=[0,2,3]) / T.sqrt(v0_sigma * vi_sigma)
autocorr_func = theano.function([v0, vi, mean], ri)


def autocorrelation(samples,
                    mean,
                    lag,
                    saving_file):


    auto_coef = numpy.zeros((samples.shape[1], lag), dtype='float32')   
    
    v0 = samples[0:lag,]
    for r in xrange(lag):    
        vr = samples[r:r+lag,]
        auto_coef[:, r] = autocorr_func(v0, vr, mean)

        if r%100 == 0:
            print r
            print auto_coef[:, r].mean()
        if r%1000 == 0:
            print "saving"
            numpy.save(filename, auto_coef)        

    numpy.save(saving_file, auto_coef)

lag = 5000 
filename = 'mnist_rbm_auto'
_mean = "%s/nips2013_mnist_rbm_exp5_1/3/rbm_samples_mean.npy" % BASE
_samples = "%s/nips2013_mnist_rbm_exp5_1/3/rbm_samples.npy" % BASE
mean = numpy.load(_mean).astype('float32').reshape(1, 28, 28)
samples = numpy.load(_samples)[:,::10].astype('float32').reshape(-1, 10, 28, 28)
autocorrelation(samples, mean, lag, filename)
del mean
del samples

filename = 'mnist_tdbn2_auto'
_mean    = "%s/nips2013_mnist_tdbn_exp5_1/10/dt_samples_mean.npy" % BASE
_samples = "%s/nips2013_mnist_tdbn_exp5_1/10/dt_samples.npy" % BASE
mean = numpy.load(_mean).astype('float32')
samples = numpy.load(_samples)[:,::10].astype('float32')
autocorrelation(samples, mean, lag, filename)
del mean
del samples

filename = 'mnist_tdbn3_auto'
_mean    = "%s/nips2013_mnist_tdbn_exp5_2/11/dt_samples_mean.npy" % BASE
_samples = "%s/nips2013_mnist_tdbn_exp5_2/11/dt_samples.npy" % BASE
mean = numpy.load(_mean).astype('float32')
samples = numpy.load(_samples)[:,::10].astype('float32')
autocorrelation(samples, mean, lag, filename)
del mean
del samples
