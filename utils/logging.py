import tables
from pylearn.io.seriestables import ErrorSeries

class HDF5Logger():

    def __init__(self, fname, to_txt=True):
        self.fname = fname
        self.to_txt = to_txt
        self.entries = {}
        self.fp = tables.openFile(fname, "w") 
        self.first_write = True

    def log(self, name, index, value, index_names=('n',)):
        if not self.entries.has_key(name):
           self.entries[name] = ErrorSeries(index_names=index_names, 
                   error_name=name, table_name=name, hdf5_file=self.fp)
        self.entries[name].append([index], value)

    def log_list(self, index, values):
        fp = open(self.fname.split('.')[0] + '.txt', 'a') if self.to_txt else None

        if self.first_write:
            fp.write('n\t')
            for (k, fmt, v) in values:
                fp.write('%s\t' % k)
            fp.write('\n')

        if fp:
            fp.write('%i\t' % index)

        for (k, fmt, v) in values:
            self.log(k, index, v)
            if fp:
                fp.write((fmt + '\t') % v)

        if fp:
            fp.write('\n')
            fp.close()
        
        self.first_write = False

    def close(self):
        self.fp.close()


