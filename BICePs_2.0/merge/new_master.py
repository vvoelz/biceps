### In BICePs 2.0, this script will play as a center role of BICePs calucltion. The users should specify any input files, type of reference potential they want to use (if other than default). --Yunhui 04/2018###

import sys, os, glob
sys.path.append('src') # source code path --Yunhui 04/2018
from Restraint import *
from PosteriorSampler import *
import cPickle  # to read/write serialized sampler classes
import argparse
import re


dataFiles = 'test_cs_H'
data = sort_data(dataFiles)
#print data
#sys.exit()

#########################################
# Let's create our ensemble of structures

if (1):
    verbose = False
#    nclusters = 50
    lam = 1.0
    energies_filename =  'energy.txt'
    energies = loadtxt(energies_filename)
    if verbose:
        print 'energies.shape', energies.shape
    energies -= energies.min()  # set ground state to zero, just in case

# We will instantiate a number of Structure() objects to construct the ensemble
ensemble = []
for i in range(energies.shape[0]):
#for i in range(2):
    print
    print '#### STRUCTURE %d ####'%i

#    expdata = loadtxt('test_cs_H/ligand1_%d.cs_H'%i)
#    data = ['test_cs_H/ligand1_%d.cs_H'%i]
    print data[i]
    s = Restraint('8690.pdb',lam,energies[i],data = data[i])

    ensemble.append( s )
#sys.exit()



  ##########################################
  # Next, let's do some posterior sampling

outdir = 'results_ref_normal'
# Temporarily placing the number of steps here...
nsteps = 1000 # 10000000
"""OUTPUT

    Files written:
        <outdir>/traj_lambda_<lambda>.yaml  - YAML Trajectory file
        <outdit>/sampler_<lambda>.pkl       - a cPickle'd sampler object
"""

# Make a new directory if we have to
if not os.path.exists(outdir):
    os.mkdir(outdir)


if (1):
    sampler = PosteriorSampler(ensemble)


    sampler.sample(nsteps)  # number of steps
    print 'Processing trajectory...',
    sampler.traj.process()  # compute averages, etc.
    print '...Done.'

    print 'Writing results...',
    sampler.traj.write_results(os.path.join(outdir,'traj_lambda%2.2f.yaml'%lam))
    print '...Done.'

    # pickle the sampler object
    print 'Pickling the sampler object ...',
    outfilename = 'sampler_lambda%2.2f.pkl'%lam
    print outfilename,
    fout = open(os.path.join(outdir, outfilename), 'wb')
    # Pickle dictionary using protocol 0.
    cPickle.dump(sampler, fout)
    print '...Done.'



