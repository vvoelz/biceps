##############################################################################
# Authors: Yunhui Ge, Rob Raddi
# This file includes functions not part of the source code but will be useful
# in different cases.
##############################################################################


##############################################################################
# Imports
##############################################################################



import sys, os, glob
import numpy as np
import re
import yaml, io

##############################################################################
# Code
##############################################################################

def sort_data(dataFiles):
    dir_list=[]
    if not os.path.exists(dataFiles):
                raise ValueError("data directory doesn't exist")
    if ',' in dataFiles:
        print 'Sorting out the data...\n'
        raw_dir = (dataFiles).split(',')
	for dirt in raw_dir:
		if dirt[-1] == '/':
			dir_list.append(dirt+'*')
		else:
			dir_list.append(dirt+'/*')
    else:
	raw_dir = dataFiles
	if raw_dir[-1] == '/':
	        dir_list.append(dataFiles+'*')
	else:
		dir_list.append(dataFiles+'/*')
#    print 'dir_list', dir_list

    data = [[] for x in xrange(7)] # list for every extension; 7 possible experimental observables supported
    # Sorting the data by extension into lists. Various directories is not an issue...
    for i in range(0,len(dir_list)):
        convert = lambda txt: int(txt) if txt.isdigit() else txt
        # This convert / sorted glob is a bit fishy... needs many tests
        for j in sorted(glob.glob(dir_list[i]),key=lambda x: [convert(s) for s in re.split("([0-9]+)",x)]):
            if j.endswith('.noe'):
                data[0].append(j)
            elif j.endswith('.J'):
                data[1].append(j)
            elif j.endswith('.cs_H'):
                data[2].append(j)
            elif j.endswith('.cs_Ha'):
                data[3].append(j)
            elif j.endswith('.cs_N'):
                data[4].append(j)
            elif j.endswith('.cs_CA'):
                data[5].append(j)
            elif j.endswith('.pf'):
                data[6].append(j)
            else:
                raise ValueError("Incompatible File extension. Use:{.noe,.J,.cs_H,.cs_Ha}")
    data = np.array(filter(None, data)) # removing any empty lists
    Data = np.stack(data, axis=-1)
    data = Data.tolist()
    return data


def list_res(input_data):
    """Determine what scheme is included in sampling"""

#    input_data = sort_data(data)
    scheme=[]
    for i in input_data[0]:
            if i.endswith('.cs_H'):
                scheme.append('cs_H')
            elif i.endswith('.cs_Ha'):
                scheme.append('cs_Ha')
            elif i.endswith('.cs_N'):
                scheme.append('cs_N')
            elif i.endswith('.cs_Ca'):
                scheme.append('cs_Ca')
            elif i.endswith('.J'):
                scheme.append('J')
            elif i.endswith('.pf'):
                scheme.append('pf')
            elif i.endswith('.noe'):
                scheme.append('noe')
                scheme.append('gamma')
            else:
                raise ValueError("Incompatible File extension. Use:{*.noe, *.J, *.cs_H, *.cs_Ha, *.cs_N, *.cs_Ca, *.pf}")

    return scheme

def write_results(self, outfilename):
    """Writes a compact file of several arrays into binary format."""

    np.savez_compressed(outfilename, self.results)

def read_results(filename):
    """Reads a npz file"""

    loaded = np.load(filename)
    print loaded.items()

def convert_pop_to_energy(pop_filename, out_filename=None):
    """Convert population to energy for each state using U = -np.log(P)"""
    if pop_filename.endwith('txt') or pop_filename.endwith('dat'):
        pop = np.loadtxt(pop_filename)
    elif pop_filename.endwith('npy'):
        pop = np.load(pop_filename)
    else:
        raise ValueError('Incompatible file extention. Use:{.txt,.dat,.npy}')
    energy=[]
    # replace NaN in the list with a very small number
    pop[np.isnan(pop)]=0.001
    for i in pop:
        energy.append(-np.log((i/float(sum(pop)))))
    if out_filename == None:
        np.savetxt('energy.txt',energy)
    else:
        np.savetxt(out_filename,energy)
