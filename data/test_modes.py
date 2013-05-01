from pylearn.datasets import Dataset
import numpy
import theano
floatX = theano.config.floatX

def neal94_AC(p=0.01, size=10000, seed=238904, w=[.25,.25,.25,.25]):
    """
    Generates the dataset used in [Desjardins et al, AISTATS 2010]. The dataset
    is composed of 4x4 binary images with four basic modes: full black, full
    white, and [black,white] and [white,black] images. Modes are created by
    drawing each pixel from the 4 basic modes with a bit-flip probability p.
    
    :param p: probability of flipping each pixel p: scalar, list (one per mode) 
    :param size: total size of the dataset
    :param seed: seed used to draw random samples
    :param w: weight of each mode within the dataset
    """

    # can modify the p-value separately for each mode
    if not isinstance(p, (list,tuple)):
        p = [p for i in w]

    rng = numpy.random.RandomState(seed)
    data = numpy.zeros((size,16))

    # mode 1: black image
    B = numpy.zeros((1,16))
    # mode 2: white image
    W = numpy.ones((1,16))
    # mode 3: white image with black stripe in left-hand side of image
    BW = numpy.ones((4,4))
    BW[:, :2] = 0
    BW = BW.reshape(1,16)
    # mode 4: white image with black stripe in right-hand side of image
    WB = numpy.zeros((4,4))
    WB[:, :2] = 1
    WB = WB.reshape(1,16)

    modes = [B,W,BW,WB]
    data = numpy.zeros((0,16))
    
    # create permutations of basic modes with bitflip prob p
    for i, m in enumerate(modes):
        n = size * w[i]
        bitflip = rng.binomial(1,p[i],size=(n,16))
        d = numpy.abs(numpy.repeat(m, n, axis=0) - bitflip)
        data = numpy.vstack((data,d))

    y = numpy.zeros((size,1))
    
    set = Dataset()
    set.train = Dataset.Obj(x=data, y=y)
    set.test = None
    set.img_shape = (4,4)

    return set

def n_modes(n_modes=4, img_shape=(4,4), size=10000,
            p=0.001, w=None, seed=238904):
    """
    Generates the dataset used in [Desjardins et al, AISTATS 2010]. The dataset
    is composed of 4x4 binary images with four basic modes: full black, full
    white, and [black,white] and [white,black] images. Modes are created by
    drawing each pixel from the 4 basic modes with a bit-flip probability p.
    
    :param p: probability of flipping each pixel p: scalar, list (one per mode) 
    :param size: total size of the dataset
    :param seed: seed used to draw random samples
    :param w: weight of each mode within the dataset
    """
    img_size = numpy.prod(img_shape)

    # can modify the p-value separately for each mode
    if not isinstance(p, (list,tuple)):
        p = [p for i in xrange(n_modes)]

    rng = numpy.random.RandomState(seed)
    data = numpy.zeros((0,img_size))

    for i, m in enumerate(range(n_modes)):
        base = rng.randint(0,2,size=(1,img_size))

        mode_size = w[i]*size if w is not None else size/numpy.float(n_modes)
        # create permutations of basic modes with bitflip prob p

        bitflip = rng.binomial(1,p[i],size=(mode_size, img_size))
        d = numpy.abs(numpy.repeat(base, mode_size, axis=0) - bitflip)
        data = numpy.vstack((data,d))

    y = numpy.zeros((size,1))
    
    set = Dataset()
    set.train = Dataset.Obj(x=data, y=y)
    set.test = None
    set.img_shape = (4,4)

    return set


class OnlineModes(object):

    def __init__(self, n_modes, img_shape, seed=238904, 
                 min_p=1e-4, max_p=1e-1,
                 min_w=0., max_w=1.,
                 w = None, p = None):

        self.n_modes = n_modes
        self.img_shape = img_shape
        self.rng = numpy.random.RandomState(seed)
        self.img_size = numpy.prod(img_shape)

        # generate random p, w values
        if p is None:
            p = min_p + self.rng.rand(n_modes) * (max_p - min_p)
        self.p = p

        if w is None:
            w = min_w + self.rng.rand(n_modes) * (max_w - min_w)
        self.w = w / numpy.sum(w)

        self.sort_w_idx = numpy.argsort(self.w)

        self.modes = self.rng.randint(0,2,size=(n_modes,self.img_size))

    def __iter__(self): return self

    def next(self, batch_size=1):

        modes = self.rng.multinomial(1, self.w, size=batch_size)
        data = numpy.zeros((batch_size, self.img_size))

        modes_i = []

        for bi, mode in enumerate(modes):
            mi, = numpy.where(mode != 0)
            modes_i.append(mi)
            bitflip = self.rng.binomial(1,self.p[mi], size=(1, self.img_size))
            data[bi] = numpy.abs(self.modes[mi] - bitflip)

        self.data = data
        self.data_modes = modes_i

        return data

class OnlineModesNIPSDLUFL(OnlineModes):

    def __init__(self):
        super(OnlineModesNIPSDLUFL, self).__init__(n_modes=5,
                img_shape=(28,28),
                w=[0.3314, 0.2262, 0.0812, 0.0254, 0.3358],
                p=[0.0001, 0.0137, 0.0215, 0.0223, 0.0544])

        ## this is the fixed testset
        self.X = numpy.zeros((1e4, 28*28))
        for i in xrange(len(self.X)):
            self.X[i] = self.next()[0]

    def get_batch_design(self, batch_size, include_labels=False):
        assert include_labels is False
        return self.next(batch_size).astype(floatX)

    def has_targets(self):
        return False
