import sys, os, glob

sys.path.append('../../src')

from Structure import *
from PosteriorSampler import *

import numpy as np

#########################################
# Let's create our ensemble of structures

expdata_filename = 'cineromycinB_expdata_VAV.yaml'
energies_filename = 'cineromycinB_QMenergies.dat'
energies = loadtxt(energies_filename)*627.509  # convert from hartrees to kcal/mol
energies = energies/0.5959   # convert to reduced free energies F = f/kT
energies -= energies.min()

# Do sampling for a series of lambdas to scale up free_energies lambda*f_i
for lam in np.arange(0.0, 1.2, 0.2):
    print 'lam =', lam

    # create an ensemble of structures with lamda-scaled free_energies
    ensemble = []
    for i in range(100):
        print '#### STRUCTURE %d ####'%i
        ensemble.append( Structure('cineromycinB_pdbs/%d.fixed.pdb'%i, lam*energies[i], expdata_filename, use_log_normal_distances=False) )

    # let's do some posterior sampling
    nsteps = 100000
    sampler = PosteriorSampler(ensemble, dlogsigma_noe=np.log(1.02), sigma_noe_min=0.7, sigma_noe_max=0.71,
                                 dlogsigma_J=np.log(1.02), sigma_J_min=5.0, sigma_J_max=5.1,
                                 dloggamma=np.log(1.01), gamma_min=1.3, gamma_max=1.31,
                                 use_reference_prior=True)
    sampler.sample(nsteps)  # number of steps
    sampler.traj.process()  # compute averages, etc.
    sampler.traj.write_results('traj_lambda_%1.1f.yaml'%lam)

    del ensemble
    del sampler


