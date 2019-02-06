import sys, os, glob

sys.path.append('../../src')

from Structure import *
from PosteriorSampler import *

from matplotlib import pyplot as plt

#########################################
# Let's create our ensemble of structures

expdata_filename = 'cineromycinB_expdata_VAV.yaml'
energies_filename = 'cineromycinB_QMenergies_PCM.dat'
energies = loadtxt(energies_filename)*627.509  # convert from hartrees to kcal/mol
energies = energies/0.5959   # convert to reduced free energies F = f/kT
energies -= energies.min()

ensemble = []
ensemble2 = []
for i in range(100):
    print
    print '#### STRUCTURE %d ####'%i
    ensemble.append( Structure('cineromycinB_pdbs/%d.fixed.pdb'%i, energies[i], expdata_filename, use_log_normal_distances=False) )
    
# collect distance distributions for each distance 
s = ensemble[0]
ndistances = len(s.distance_restraints)
all_distances = []
distance_distributions = [[] for i in range(ndistances)]
for s in ensemble:
    for i in range(len(s.distance_restraints)):
        distance_distributions[i].append( s.distance_restraints[i].model_distance )
        all_distances.append( s.distance_restraints[i].model_distance )

# compile names of each distance
raw = """- [23, 25]    0    H(3)/H(6)
- [23, 27]    1    H(3)/H(9)
- [25, 27]    2    H(6)/H(9) 
- [24, 26]    3    H(5)/H(7)
- [25, 26]    4    H(6)/H(7)
- [26, 27]    5    H(7)/H(9) 
- [27, 32]    6    H(9)/H(12)
- [27, 30]    7    H(9)/H(11') a
- [27, 31]    7    H(9)/H(11') b
- [28, 38]    8    Me(12)/H(10') a
- [28, 39]    8    Me(12)/H(10') b
- [28, 40]    8    Me(12)/H(10') c
- [29, 38]    8    Me(12)/H(10') d
- [29, 39]    8    Me(12)/H(10') e
- [29, 40]    8    Me(12)/H(10') f
- [25, 41]    9    Me(4)/H(6) a
- [25, 42]    9    Me(4)/H(6) b
- [25, 43]    9    Me(4)/H(6) c
- [26, 44]    10   Me(8)/H(7) a
- [26, 45]    10   Me(8)/H(7) b
- [26, 46]    10   Me(8)/H(7) c
- [33, 38]    11   Me(12)/H(13) a
- [33, 39]    11   Me(12)/H(13) b
- [33, 40]    11   Me(12)/H(13) c
- [35, 38]    12   Me(12)/Me(13) a
- [35, 39]    12   Me(12)/Me(13) b
- [35, 40]    12   Me(12)/Me(13) c
- [36, 38]    12   Me(12)/Me(13) d
- [36, 39]    12   Me(12)/Me(13) e
- [36, 40]    12   Me(12)/Me(13) f  
- [37, 38]    12   Me(12)/Me(13) g
- [37, 39]    12   Me(12)/Me(13) h
- [37, 40]    12   Me(12)/Me(13) i"""

names = []
rawlines = raw.split('\n')
for line in rawlines:
    names.append( line[19:] )


if (1):
    # Make a plot of the distribution of the many distances 
    plt.figure()

    print 'ndistances', ndistances # 33
    for i in range(ndistances):
        plt.subplot(11,3,i+1)
        # plot the distribution of distances across all structures
        values, bins = np.histogram(distance_distributions[i], bins=np.arange(0,10.,0.1), normed=True )
        plt.step(bins[0:-1], values)
        # plot the maximum likelihood exponential distribution fitting the data
        beta = np.array(distance_distributions[i]).sum()/(len(distance_distributions[i])+1.0)
        print 'distance', i, 'beta', beta
        tau = (1.0/beta)*np.exp(-bins[0:-1]/beta)
        plt.plot(bins[0:-1], 10*tau, 'k-')
        plt.plot([beta, beta], [0, values.max()], 'r-')
        plt.xlim(0,6)
        plt.ylim(0,3)
        plt.yticks([])
        if (i < 30):
            plt.xticks([])
        else:
            plt.xlabel("d (A)")
        plt.text(0.1, 0.5, names[i])
    #plt.tight_layout()  # NO!
    plt.show()

