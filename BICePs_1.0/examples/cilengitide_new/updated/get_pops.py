import os, sys, glob
from msmbuilder.msm_analysis import io

from scipy import loadtxt, savetxt, savez
import numpy as np

usage = "Usage:  python get_pops.py"

assign = io.loadh('Assignment100.h5')
print assign

nclusters = assign['arr_0'].max() + 1
print 'nclusters', nclusters

pops, bin_edges = np.histogram(assign['arr_0'], bins=range(nclusters+1), normed=True)
print pops, bin_edges
print 'pops.shape', pops.shape

# save a list of unsorted populations
savetxt('populations.dat', pops)

# save a list of free energies (in units kT)
f = -np.log(pops)
f -= f.min()
savetxt('reduced_free_energies.dat', f)


# sort the populations
Ind = np.argsort(-pops)
print 'State\tpopulation'
array_of_pops=[]
for i in Ind:
    print '%d\t%f'%(i, pops[i])
    array_of_pops.append(pops[i])

savetxt('populations_sorted.dat',array_of_pops)
