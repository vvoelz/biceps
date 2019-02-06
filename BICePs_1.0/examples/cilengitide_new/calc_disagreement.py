import sys, os, glob

sys.path.append('../../src')

from Structure import *
from PosteriorSampler import *

from matplotlib import pyplot as plt
from scipy import loadtxt, savetxt

import cPickle, pprint

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("samplerfile", help="the name of the cPickle'd sampler object")
parser.add_argument("popsfile", help="the corresponsing filename containing state populations and their uncertainties")
parser.add_argument("gamma", help="the value of the scaling parameter gamma_QMexp = 1.249, gamma_exp = 1.379", type=float)
parser.add_argument("-v", "--verbose", help="print verbosely", action="store_true")
args = parser.parse_args()

print '=== Settings ==='
print 'samplerfile', args.samplerfile
print 'popsfile', args.popsfile
print 'gamma', args.gamma
print 'verbose', args.verbose

# Load sampler object
if args.verbose:
    print 'Loading %s ...'%args.samplerfile
sampler = cPickle.load( open(args.samplerfile, 'rb') ) 

# Load populations
P_dP = loadtxt(args.popsfile)
K = P_dP.shape[1]/2 
if args.samplerfile.count('1.00') > 0:
    pops = P_dP[:,K-1]   # assume lambda=1.00 is the last column in each half
else:
    pops = P_dP[:,0]   # assume lambda=0.00

#print 'pops', pops

# collect distance distributions for each distance 
ensemble = sampler.ensembles[0]
disagreement = 0.0
disagreement_J = 0.0

for i in range(len(ensemble)):
    s = ensemble[i]
    this_disagreement = 0.0
    this_disagreement_J = 0.0
    N = 0.0
    N_J = 0.0
    for j in range(len(s.distance_restraints)):
        if args.verbose:
            print j, 'model_distance', s.distance_restraints[j].model_distance,
            print 'exp_distance*args.gamma', s.distance_restraints[j].exp_distance*args.gamma
        this_disagreement += s.distance_restraints[j].weight*np.abs( s.distance_restraints[j].model_distance - s.distance_restraints[j].exp_distance*args.gamma )
        N += s.distance_restraints[j].weight
    if args.verbose:
        print i, pops[i], this_disagreement/N
    disagreement += pops[i]*this_disagreement/N

    for j in range(len(s.dihedral_restraints)):
        if args.verbose:
            print j, 'model_Jcoupling', s.dihedral_restraints[j].model_Jcoupling,
            print 'exp_Jcoupling', s.dihedral_restraints[j].exp_Jcoupling
        this_disagreement_J += s.dihedral_restraints[j].weight*np.abs( s.dihedral_restraints[j].model_Jcoupling - s.dihedral_restraints[j].exp_Jcoupling )
        N_J += s.dihedral_restraints[j].weight
    if args.verbose:
        print i, pops[i], this_disagreement_J/N_J
    disagreement_J += pops[i]*this_disagreement_J/N_J

print 'average distance disagreement =', disagreement 
print 'average Jcoupling disagreement =', disagreement_J

if (1):
    # Make a plot of the distribution of the many distances 
    plt.figure()

    ndistances = 33
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

