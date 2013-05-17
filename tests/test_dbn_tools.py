import numpy

from deep_tempering import rbm
from deep_tempering import tempered_dbn
from deep_tempering.scripts.likelihood import rbm_tools
from deep_tempering.scripts.likelihood import dbn_tools

def build_dbn():

    lrdef = {'type': 'linear', 'start': 0.001, 'end': 0.001}
    spwdef = {'h': 0}
    sptdef = {'h': 0.1}
    iscales = {'W': 0.01, 'hbias':0, 'vbias':0}
    flags = {'enable_centering': 0}

    rbm0 = rbm.RBM(n_v=10, n_h=10, iscales=iscales, lr_spec=lrdef,
            sp_weight=spwdef, sp_targ=sptdef, flags=flags)
    rbm0.Wv.set_value(numpy.load('W0.npy'))
    rbm0.vbias.set_value(numpy.load('b0.npy').flatten())
    rbm0.hbias.set_value(numpy.load('c0.npy').flatten())

    rbm1 = rbm.RBM(n_v=10, n_h=20, iscales=iscales, lr_spec=lrdef,
            sp_weight=spwdef, sp_targ=sptdef, flags=flags)
    rbm1.Wv.set_value(numpy.load('W1.npy'))
    rbm1.vbias.set_value(numpy.load('b1.npy').flatten())
    rbm1.hbias.set_value(numpy.load('c1.npy').flatten())
    
    dbn = tempered_dbn.TemperedDBN(rbms=[rbm0,rbm1])

    # load test data
    samples = numpy.load('samples.npy')

    # Estimate DBN lower-bound
    rbm1_logz = rbm_tools.compute_log_z(rbm1, rbm1.fe_v_func)

    #(ais_logz, var_logz), _aisobj = rbm_tools.rbm_ais(
                        #rbm1.get_uncentered_param_values(),
                        #n_runs=100,
                        #betas=numpy.sqrt(numpy.arange(0, 1, 0.001)))

    lbound = dbn_tools.compute_likelihood_lbound(dbn, rbm1_logz, samples)
    print '*** Ruslan bound ***'
    print 'rbm1_logz = %f' % rbm1_logz
    print 'DBN lbound is %f' % lbound

    lbound = dbn_tools.compute_likelihood_lbound_theis(dbn, rbm1_logz, samples)
    print '*** Theis bound ***'
    print 'rbm1_logz = %f' % rbm1_logz
    print 'DBN lbound is %f' % lbound

    print """
    *** Theis code ***
    ais_logz =  13.300605056
    brf_logz =  13.300605056
    dbn_bound =  -8.11698469093
    dbn_probs =  -6.95490613962
    brf_probs =  -6.93064060165
    """

if __name__ == '__main__':
    build_dbn()
