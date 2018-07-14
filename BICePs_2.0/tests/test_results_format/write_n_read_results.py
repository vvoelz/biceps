
import numpy as np
import yaml, io
import h5py
import pickle
import xml


# Numpy Z Compression
#NOTE: This will work well with Cython if we go that route.
# Standardized: Yes ; Binary: Yes; Human Readable: No;

def write_results(self, outfilename):
    """Writes a compact file of several arrays into binary format."""

    np.savez_compressed(outfilename, self.results)

def read_results(filename):
    """Reads a npz file"""

    loaded = np.load(filename)
    print loaded.items()


# YAML
#NOTE:
# Standardized: Yes; Binary: No; Human Readable: Yes;

def write_results(self, outfilename):
    """Dumps results to a YAML format file. """

    fout = file(outfilename, 'w')
    yaml.dump(self.results, fout, default_flow_style=False)

def read_results(filename):
    '''Reads a yaml file.'''

    with io.open(filename,'r') as file:
        loaded_data = yaml.load(file)
        print('%s'%loaded_data).replace(" '","\n\n '")


# H5
#NOTE: Cython wrapping of the HDF5 C API
# Standardized: Yes; Binary: Yes; Human Readable: No;

def write_results(self, outfilename):
    """ """

    hf = h5py.File(outfilename, 'w')
    hf.create_dataset('dataset', data=self.results)
    hf.close()


def read_results(filename):
    f = h5py.File(filename,'r')
    print f.items()
    #a_group_key = list(f.keys())[0]
    # Get the data
    #data = list(f[a_group_key])


# Python Pickle
#NOTE:
# Standardized: Yes; Binary: Yes; Human Readable: No;

def write_results(self, outfilename):
    """Writes results as a pickle file."""

    pkl = open(outfilename, 'wb')
    pickle.dump(self.results, pkl)
    pkl.close()

def read_results(filename):
    """ """

    pkl = open(filename, 'r')
    loaded = pickle.load(pkl)
    print loaded





# xml
#NOTE: ? Using minidom to parse XML and Element Tree to process XML ?
# Standardized: Yes; Binary: Partial; Human Readable: Yes;

def write_results(self, outfilename='traj.h5'):
    """Writes results in xml file format. """
    pass


def read_results(filename):
    pass












# Testing to ensure the code works
#
#filetype  = ['npz','yaml','h5','pkl']
#filename = 'test.'+filetype[3]
#x = np.random.rand(3, 2)
#results = x
#write_results(results,outfilename=filename)
#read_results(filename)
#
#def write_results(results, outfilename):
#    """Writes results as a pickle file."""
#
#    pkl = open(outfilename, 'wb')
#    pickle.dump(results, pkl)
#    pkl.close()
#
#def read_results(filename):
#    """ """
#
#    pkl = open(filename, 'r')
#    loaded = pickle.load(pkl)
#    print loaded












