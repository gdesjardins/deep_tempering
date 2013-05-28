import numpy
import os
import time

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

    def __init__(self, which_set, one_hot = False):
        """
        :param which_set: one of ['train','test']
        :param center: data is in range [0,256], center=True subtracts 127.5.
        :param multi_target: load extra information as additional labels.
        """
        assert which_set in ['train','test']
        self.which_set = which_set

        # Load data and labels.
        base = '%s/norb_small/ruslan_binarized' % os.getenv('PYLEARN2_DATA_PATH')
        fname = '%s/norb96x96x2_fov8422_%s_X.npy' % (base, which_set)
        X = numpy.load(fname)
        fname = '%s/norb96x96x2_fov8422_%s_Y.npy' % (base, which_set)
        y = numpy.load(fname).astype('int')

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

    def next(self):
        fovx = self.iterator.next()

        t1 = time.time()
        new_fovx = shift.shift_batch(
                fovx,
                topo_shape = self.topo_shape,
                rings = self.rings,
                maxshift = self.max_shift,
                rng = self.rng)
        #print 'Shift: ', time.time() - t1

        t1 = time.time()
        bin_fovx = self.grbm.preproc(new_fovx)
        #print 'GRBM: ', time.time() - t1
        return bin_fovx


class TrainingAlgorithm(default.DefaultTrainingAlgorithm):

    def setup(self, model, dataset):

        it = dataset.iterator(mode='sequential',
                batch_size = model.batch_size)
        dataset._iterator = PreprocIterator(it,
                (96,96,2), (8,4,2,2), 6)

        x = dataset._iterator.next()
        model.init_parameters_from_data(x)
 
        super(TrainingAlgorithm, self).setup(model, dataset)
