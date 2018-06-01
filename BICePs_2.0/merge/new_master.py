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

# Old Structure:{{{
#    s = Structure('8690.pdb', args.lam*energies[i],data = data,
#            use_log_normal_distances=False,
#            dloggamma=np.log(1.01), gamma_min=0.2, gamma_max=5.0)
#
# }}}

    # NOTE: Upon instantiation, each Structure() object computes the distances from the given PDB.
    #       However, our clusters represent averaged conformation states, and so we
    #       need to replace these values with our own r^-6-averaged, precomputed ones

    # replace PDB distances with r^-6-averaged distances
#    print 'len(s.distance_restraints)', len(s.distance_restraints)
#    for j in range(len(s.distance_restraints)):
#        print s.distance_restraints[j].i, s.distance_restraints[j].j, model_distances[j]
#        s.distance_restraints[j].model_distance = model_distances[j]
#	print s.distance_restraints[j].i, s.distance_restraints[j].j, s.distance_restraints[j].exp_distance
#    for j in range(len(r_cs_H.chemicalshift_H_restraints)):
#        print s.chemicalshift_H_restraints[j].i, model_chemicalshift_H[j]
#        s.chemicalshift_H_restraints[j].model_chemicalshift_H = model_chemicalshift_H[j]


    # update the distance sse's!
#    s.compute_sse_chemicalshift_H()

    # update the protectionfactor sse's!                #GYH
   # s.compute_sse_protectionfactor()
#    print 's.sse_cs_H', s.sse_cs_H
#    print 's.sse_distances', s.sse_distances
    # add the structure to the ensemble
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

    # Old Sampler:{{{
#    sampler = PosteriorSampler(ensemble, data=data,
#          dlogsigma_noe=np.log(1.01), sigma_noe_min=0.2, sigma_noe_max=5.0,
#          dlogsigma_J=np.log(1.02), sigma_J_min=0.05, sigma_J_max=20.0,
#          dlogsigma_cs_H=np.log(1.02), sigma_cs_H_min=0.01, sigma_cs_H_max=5.0,
#          dlogsigma_cs_Ha=np.log(1.02), sigma_cs_Ha_min=0.01, sigma_cs_Ha_max=5.0,
#	  dlogsigma_cs_N=np.log(1.01), sigma_cs_N_min=0.01, sigma_cs_N_max=10.0,
#          dlogsigma_cs_Ca=np.log(1.01), sigma_cs_Ca_min=0.01, sigma_cs_Ca_max=10.0,
#          dlogsigma_PF=np.log(1.01), sigma_PF_min=0.01, sigma_PF_max=10.0,               #GYH
#          dloggamma=np.log(1.01), gamma_min=0.2, gamma_max=5.0,
#          dalpha=0.1, alpha_min=0.0, alpha_max=0.1,
#          ambiguous_groups=False, # RMR
#          sample_ambiguous_distances=False,
#          distribution='exponential',#distribution='gaussian'
#          use_reference_prior_H=True,
#          use_reference_prior_noe=False, use_reference_prior_H=True,
#          use_reference_prior_Ha=False, use_reference_prior_N=False,
#          use_reference_prior_Ca=False, use_reference_prior_PF=False,
#          use_gaussian_reference_prior_noe=False,
#          use_gaussian_reference_prior_H=False,
#          use_gaussian_reference_prior_Ha=False,
#          use_gaussian_reference_prior_N=False,
#          use_gaussian_reference_prior_Ca=False,
#          use_gaussian_reference_prior_PF=False)
  #sampler = PosteriorSampler(ensemble, use_reference_prior=True)
# }}}


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

#}}}


