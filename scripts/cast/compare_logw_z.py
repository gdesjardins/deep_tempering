import tables
import pickle
import numpy
import theano
import theano.tensor as T
from deep_tempering.scripts.likelihood import rbm_tools
floatX = theano.config.floatX

data = tables.openFile('cast_callback.hdf5')
logw = [data.getNode('/beta_logw').col('%02d' % i)[-1] for i in xrange(100)]

model = pickle.load(open('model_e530000.pkl'))
b = theano.shared(numpy.ones(1).astype(floatX), name='b')
x = T.matrix('x')
fe = theano.function([x], model.free_energy_h(x, beta=b))

betas = numpy.linspace(1.0, 0., 100)
zbs = numpy.zeros(len(betas))
for i, beta in enumerate(betas):
    b.set_value(numpy.float32(beta * numpy.ones(1)))
    zbs[i] = rbm_tools.compute_log_z(model, fe)

print 'log z(beta_k) = ', zbs
print 'g_k = ', logw
print 'g_k + z.min()  = ', logw + zbs.min()

import pylab as pl
pl.plot(zbs / numpy.sum(zbs), label='Z')
logw += zbs.min()
pl.plot(logw / numpy.sum(logw), label='logw')
pl.legend()
pl.show()
