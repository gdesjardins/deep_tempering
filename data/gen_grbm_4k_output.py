import numpy
from deep_tempering.data.grbm_preproc import GRBMPreprocessor
from deep_tempering.data import binary_norb

if __name__ == '__main__':

    grbm = GRBMPreprocessor()

    for which_set in ['train', 'test']:
        data = binary_norb.FoveatedPreprocNORB(which_set)
        out = numpy.zeros((len(data), grbm.Wv.shape[1]))
        import pdb; pdb.set_trace()
        for i in xrange(0, len(data.X), 1000):
            batch = data.X[i:i+1000]
            out[i:i+len(batch)] = grbm.preproc(batch)
        base = '%s/norb_small/ruslan_binarized' % os.getenv('PYLEARN2_DATA_PATH')
        fname = '%s/norb96x96x2_fov8422_grbm_4k_%s_X.npy' % (base, which_set)
        numpy.save(fname, out)
        del data, out
