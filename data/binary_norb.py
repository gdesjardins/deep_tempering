import numpy
import os
import time
import copy

from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets import retina

from pylearn2.training_algorithms import default
from deep_tempering.data import shift
from deep_tempering.data.grbm_preproc import GRBMPreprocessor

def onehot_encoding(y):
    one_hot = numpy.zeros((y.shape[0],5),dtype='float32')
    for i in xrange(y.shape[0]):
        one_hot[i,y[i]] = 1
    return one_hot

class BinaryNORB(dense_design_matrix.DenseDesignMatrix):

    def __init__(self, which_set, one_hot = False):
        """
        :param which_set: one of ['train','test']
        """
        assert which_set in ['train','test']
        self.which_set = which_set

        # Load data and labels.
        base = '%s/norb_small/ruslan_binarized' % os.getenv('PYLEARN2_DATA_PATH')
        fname = '%s/%s_X.npy' % (base, which_set)
        X = numpy.load(fname)
        fname = '%s/%s_Y.npy' % (base, which_set)
        y = numpy.load(fname).astype('int')

        self.one_hot = one_hot
        if one_hot:
            y = onehot_encoding(y)

        super(BinaryNORB, self).__init__(X = X, y = y)

class NumpyLoader(dense_design_matrix.DenseDesignMatrix):

    def __init__(self, fname):
        """
        :param which_set: one of ['train','test']
        """
        self.which_set = fname.split('.')[0]
        # Load data and labels.
        base = '%s/norb_small/ruslan_binarized' % os.getenv('PYLEARN2_DATA_PATH')
        fname = '%s/%s' % (base, fname)
        X = numpy.load(fname)
        y = numpy.zeros(X.shape[0])
        super(NumpyLoader, self).__init__(X = X, y = y)


class MyBinaryNORB(dense_design_matrix.DenseDesignMatrix):

    def __init__(self, which_set, one_hot = False):
        """
        :param which_set: one of ['train','test']
        """
        assert which_set in ['train','test']
        self.which_set = which_set

        # Load data and labels.
        base = '%s/norb_small/ruslan_binarized' % os.getenv('PYLEARN2_DATA_PATH')
        fname = '%s/norb96x96x2_fov8422_grbm_4k_%s_X.npy' % (base, which_set)
        X = numpy.load(fname)
        fname = '%s/%s_Y.npy' % (base, which_set)
        y = numpy.load(fname).astype('int')

        self.one_hot = one_hot
        if one_hot:
            y = onehot_encoding(y)

        super(MyBinaryNORB, self).__init__(X = X, y = y)


class FoveatedPreprocNORB(dense_design_matrix.DenseDesignMatrix):
    """
    This dataset can serve two purposes.
    
    When used by itself, it loads up the preprocessed and foveated NORB data, used to train the
    first layer GRBM (model used to binarize the dataset).
    
    When used in conjunction with binary_norb.TrainingAlgorithm, will generate binarized
    (through a GRBM) shifted version of this foveated NORB dataset. This thus generates a
    binary representation (online) which can be used with binary RBMs or DBMs.
    """

    def __init__(self, which_set, one_hot = False, seed=1239):
        """
        :param which_set: one of ['train', 'valid', 'test']
        :param center: data is in range [0,256], center=True subtracts 127.5.
        :param multi_target: load extra information as additional labels.
        """
        assert which_set in ['train', 'valid', 'test']
        self.which_set = which_set

        # Load data and labels.
        base = '%s/norb_small/ruslan_binarized' % os.getenv('PYLEARN2_DATA_PATH')
        if which_set in ['train', 'valid']:
            xfname = '%s/norb96x96x2_fov8422_%s_X.npy' % (base, 'train')
            yfname = '%s/norb96x96x2_fov8422_%s_Y.npy' % (base, 'train')
        else:
            xfname = '%s/norb96x96x2_fov8422_%s_X.npy' % (base, which_set)
            yfname = '%s/norb96x96x2_fov8422_%s_Y.npy' % (base, which_set)

        X = numpy.load(xfname)
        y = numpy.load(yfname).astype('int')

        if which_set in ['train', 'valid']:
            rng = numpy.random.RandomState(seed)
            pidx = rng.permutation(len(X))
            idx = pidx[:-4300] if which_set == 'train' else pidx[-4300:]
            X = X[idx]
            y = y[idx]

        self.one_hot = one_hot
        if one_hot:
            y = onehot_encoding(y)

        view_converter = retina.RetinaCodingViewConverter((96,96,2), (8,4,2,2))

        super(FoveatedPreprocNORB,self).__init__(X = X, y = y, view_converter = view_converter)


class PreprocIterator():
    """
    A basic iterator which fetches the next example in the dataset, and then performs a random
    shift (as described in the tempered transition paper).
    """

    def __init__(self, iterator, topo_shape, rings, max_shift, seed=129387):
        """
        :param iterator: an iterator which loops over the "raw" (foveated, unjitted,
        unbinarized) NORB dataset
        """
        self.topo_shape = topo_shape
        self.rings = rings
        self.max_shift = max_shift
        self.rng = numpy.random.RandomState(seed)
        self.grbm = GRBMPreprocessor()
        # encapsulate the behavior of a "normal" dataset iterator
        self.iterator = iterator
        self._subset_iterator = iterator._subset_iterator

    def __iter__(self):
        return self

    def debug(self, fx):

        # Unfoveated the current batch
        fx1 = copy.copy(fx)
        x1  = retina.decode(fx1, (96,96,2), (8,4,2,2))
        
        # Binarized, Reconstruct then defoveate minibatch
        fx2  = copy.copy(fx)
        bfx2 = self.grbm.preproc(fx2)
        fxhat2 = self.grbm.reconstruct(bfx2)
        xhat2  = retina.decode(fxhat2, (96,96,2), (8,4,2,2))

        # Shift then defoveate minibatch
        fx3  = copy.copy(fx)
        sfx3 = shift.shift_batch(fx3,
                topo_shape = self.topo_shape,
                rings = self.rings,
                maxshift = self.max_shift,
                rng = self.rng)
        sx3  = retina.decode(sfx3, (96,96,2), (8,4,2,2))

        # Shift, binarize, reconstruct, then defoveate minibatch
        bsfx4 = self.grbm.preproc(sfx3)
        sfxhat4 = self.grbm.reconstruct(bsfx4)
        sxhat4  = retina.decode(sfxhat4, (96,96,2), (8,4,2,2))
 
        import pylab as pl
        import pdb; pdb.set_trace()

        for i in xrange(len(fx)):
            pl.subplot(1,4,1); pl.gray(); pl.imshow(x1[i,:,:,0])
            pl.subplot(1,4,2); pl.gray(); pl.imshow(sx3[i,:,:,0])
            pl.subplot(1,4,3); pl.gray(); pl.imshow(xhat2[i,:,:,0])
            pl.subplot(1,4,4); pl.gray(); pl.imshow(sxhat4[i,:,:,0])
            pl.show()

        return bin_fovx


    def next(self, debug=False):
        _fovx = self.iterator.next()

        # make explicit copy of batch data so we don't overwrite the original example !
        fovx = copy.copy(_fovx)

        # Shift then defoveate minibatch
        shift.shift_batch(fovx,
                topo_shape = self.topo_shape,
                rings = self.rings,
                maxshift = self.max_shift,
                rng = self.rng)

        bin_shift_fovx = self.grbm.preproc(fovx)

        return bin_shift_fovx


class TrainingAlgorithm(default.DefaultTrainingAlgorithm):

    def setup(self, model, dataset):

        dataset._iterator = PreprocIterator(
                dataset.iterator(
                    mode='shuffled_sequential',
                    batch_size = model.batch_size),
                topo_shape = (96,96,2),
                rings = (8,4,2,2),
                max_shift = 6)

        x = dataset._iterator.next()
        model.init_parameters_from_data(x)
 
        super(TrainingAlgorithm, self).setup(model, dataset)


if __name__ == '__main__':

    """
    from deep_tempering.data import grbm_preproc
    grbm = grbm_preproc.GRBMPreprocessor()
    # binary data extracted from Russ' MATLAB code (batchdata in MATLAB)
    binrusX1 = binary_norb.BinaryNORB('train')
    # foveated & other preprocessing's on NORB (fovimg1 and fovimg2 in MATLAB)
    fovrusX2 = binary_norb.FoveatedPreprocNORB('train')
    binrusX2 = grbm.encode(fovrusX2.X)
    # We need to find the random mapping which was used to build "batchdata", so that we can
    # numpy.sum(x1**2, axis=1)[:,None] + numpy.sum(x2**2, axis=1)[None,:] - 2*numpy.dot(x1, x2.T)
    """

    from deep_tempering.data import grbm_preproc
    grbm = grbm_preproc.GRBMPreprocessor()

    # generate a static validation and test set for the callback methods to work with
    train = FoveatedPreprocNORB('train')
    binary_train = grbm.preproc(train.X)
    numpy.save('/data/lisa/data/norb_small/ruslan_binarized/binary_train_GD.npy', binary_train)
    del train, binary_train

    # generate a static validation and test set for the callback methods to work with
    valid = FoveatedPreprocNORB('valid')
    binary_valid = grbm.preproc(valid.X)
    numpy.save('/data/lisa/data/norb_small/ruslan_binarized/binary_valid_GD.npy', binary_valid)
    del valid, binary_valid

    # generate a static validation and test set for the callback methods to work with
    test = FoveatedPreprocNORB('test')
    binary_test = grbm.preproc(test.X)
    numpy.save('/data/lisa/data/norb_small/ruslan_binarized/binary_test_GD.npy', binary_test)
    del test, binary_test

