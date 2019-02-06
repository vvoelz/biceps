import sys, os, glob

sys.path.append('../../src')

from Structure import *
from PosteriorSampler import *

from matplotlib import pyplot as plt

#########################################
# Let's create our ensemble of structures

expdata_filename = 'cineromycinB_expdata_VAV.yaml'
energies_filename = 'cineromycinB_QMenergies.dat'
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
        plt.xlabel("d (A)")
        plt.ylim(0,3)
        #plt.ylabel("P(d)")
        plt.yticks([])
        if (i < 30):
            plt.xticks([])
    plt.show()

