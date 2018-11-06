import sys, os, glob
import numpy as np
sys.path.append('src/')
from Structure import *
from PosteriorSampler import *

import cPickle  # to read/write serialized sampler classes

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("lam", help="a lambda value between 0.0 and 1.0  denoting the Hamiltonian weight (E_data + lambda*E_QM)", type=float)
parser.add_argument("outdir", help="the name of the output directory")
parser.add_argument("nsteps", help="Number of sampling steps", type=int)
parser.add_argument("--noref", help="Do not use reference potentials (default is to use them)",
                    action="store_true")
parser.add_argument("--lognormal", help="Use log-normal distance restraints (default is normal)",
                    action="store_true")
parser.add_argument("--verbose", help="use verbose output",
                    action="store_true")
args = parser.parse_args()


print '=== Settings ==='
print 'lam', args.lam
print 'outdir', args.outdir
print 'nsteps', args.nsteps
print '--noref', args.noref
print '--lognormal', args.lognormal
print '--verbose', args.verbose



"""OUTPUT 

    Files written:
        <outdir>/traj_lambda_<lambda>.yaml  - YAML Trajectory file 
        <outdit>/sampler_<lambda>.pkl       - a cPickle'd sampler object
"""

# Make a new directory if we have to
if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)


#########################################
# Let's create our ensemble of structures

# experimental restraints 
expdata_filename_noe = 'H030.noe' 
#expdata_filename_J = ''
#expdata_filename_cs_H = 'trploop2a_H.chemicalshift'
#expdata_filename_cs_Ha = 'trploop2a_Ha.chemicalshift'
# model energies
if (1):
    nclusters = len(np.loadtxt("energy.txt"))
    energies_filename = 'energy.txt'
    energies = loadtxt(energies_filename)
 #   print 'energies.shape', energies.shape
    energies -= energies.min()  # set ground state to zero, just in case


#if (1):
    # model distances 
#    model_distances = loadtxt('NOE/rminus6_whole_state0.txt')  #GYH:We have rminus6 data already
#    print 'model_distances.shape', model_distances.shape
#    print 'model_distances', model_distances

############

# We will instantiate a number of Structure() objects to construct the ensemble
ensemble = []

for i in range(nclusters):

    print
    print '#### STRUCTURE %d ####'%i

    # no information from QM --> lam = 0.0
    # QM + exp               --> lam = 1.0
    ## s = Structure('gens-pdb-kcenters-dih-1.8/Gen%d.pdb'%i, args.lam*energies[i], expdata_filename, use_log_normal_distances=False)
    model_distances = loadtxt('NOE/rminus6_whole_state%d.txt'%i)*10.0 # convert to A
#    model_chemicalshift_H = loadtxt('chemical_shift/cs/H/cs_%d.txt'%i)
#    model_chemicalshift_Ha = loadtxt('chemical_shift/cs/Ha/cs_%d.txt'%i)
#    s = Structure('Gens/Gens%d.pdb'%i, args.lam*energies[i], expdata_filename_noe, expdata_filename_J, expdata_filename_cs, use_log_normal_distances=False, dloggamma=np.log(1.01), gamma_min=0.2, gamma_max=5.0)
    s = Structure('pdb/state_%d.pdb'%i, args.lam*energies[i], expdata_filename_noe=expdata_filename_noe, use_log_normal_distances=False, dloggamma=np.log(1.01), gamma_min=0.2, gamma_max=5.0, dalpha=0.1, alpha_min=0.0, alpha_max=0.1)
#    s = Structure('Gens/Gens%d.pdb'%i, args.lam*energies[i],  expdata_filename_cs_H=expdata_filename_cs_H, expdata_filename_cs_Ha=expdata_filename_cs_Ha,  use_log_normal_distances=False, dloggamma=np.log(1.01), gamma_min=0.2, gamma_max=5.0, dalpha=0.1, alpha_min=0.0, alpha_max=0.1)
#    s = Structure('Gens/Gens%d.pdb'%i, args.lam*energies[i], expdata_filename_noe=expdata_filename_noe, use_log_normal_distances=False, dloggamma=np.log(1.01), gamma_min=0.2, gamma_max=5.0)

    # NOTE: Upon instantiation, each Structure() object computes the distances from the given PDB.
    #       However, our clusters represent averaged conformation states, and so we   
    #       need to replace these values with our own r^-6-averaged, precomputed ones

    # replace PDB distances with r^-6-averaged distances
#    print 'len(s.distance_restraints)', len(s.distance_restraints)
    for j in range(len(s.distance_restraints)):
#        print s.distance_restraints[j].i, s.distance_restraints[j].j, model_distances[j]
        s.distance_restraints[j].model_distance = model_distances[j]

    # replace chemicalshift  with  precomputed ones
#    print 'len(s.chemicalshift_H_restraints)', len(s.chemicalshift_H_restraints)
#    for j in range(len(s.chemicalshift_H_restraints)):
#        print s.chemicalshift_H_restraints[j].i, model_chemicalshift_H[j]
#        s.chemicalshift_H_restraints[j].model_chemicalshift_H = model_chemicalshift_H[j]

    # replace chemicalshift  with  precomputed ones
#    print 'len(s.chemicalshift_Ha_restraints)', len(s.chemicalshift_Ha_restraints)
#    for j in range(len(s.chemicalshift_Ha_restraints)):
#        print s.chemicalshift_Ha_restraints[j].i, model_chemicalshift_Ha[j]
#        s.chemicalshift_Ha_restraints[j].model_chemicalshift_Ha = model_chemicalshift_Ha[j]


    # update the distance sse's!
#    s.compute_sse_chemicalshift_H()
#    s.compute_sse_chemicalshift_Ha()


    # update the distance sse's!
    s.compute_sse_distances()

    # update the protectionfactor sse's!                #GYH
    s.compute_sse_protectionfactor()

    # add the structure to the ensemble
    ensemble.append( s )




# Print out the agreement for model 53 (highest-pop)
#for drest in ensemble[53].distance_restraints:
#    print 'state 53 d[ %d - %d ] ='%(drest.i, drest.j), drest.model_distance, 'd_exp =', drest.exp_distance

#model_distances = [drest.model_distance for drest in ensemble[53].distance_restraints]
#exp_distances = [drest.exp_distance for drest in ensemble[53].distance_restraints]

if (0):
  plt.figure()
  lookat = [51, 53, 55, 0]
  for i in range(len(lookat)):
    model_distances = [drest.model_distance for drest in ensemble[lookat[i]].distance_restraints]
    exp_distances = [drest.exp_distance for drest in ensemble[lookat[i]].distance_restraints]
    print 'model %d'%lookat[i], exp_distances, 'model_distances', model_distances
    plt.subplot(2,2,i+1) 
    plt.plot(exp_distances, model_distances,'.')
    plt.plot([0,5],[0,5],'k-')
    plt.xlabel('exp distance ($\\AA$)')
    plt.ylabel('model distance ($\\AA$)')
    plt.title('model %d'%lookat[i])
  plt.show()


  ##########################################
  # Next, let's do some posterior sampling


else:
  #sampler = PosteriorSampler(ensemble, use_reference_prior=True, sample_ambiguous_distances=False)
  sampler = PosteriorSampler(ensemble, dlogsigma_noe=np.log(1.01), sigma_noe_min=0.2, sigma_noe_max=10.0,
                                 dlogsigma_J=np.log(1.02), sigma_J_min=0.05, sigma_J_max=20.0,
                                 dlogsigma_cs_H=np.log(1.02), sigma_cs_H_min=0.01, sigma_cs_H_max=5.0,
                                 dlogsigma_cs_Ha=np.log(1.02), sigma_cs_Ha_min=0.01, sigma_cs_Ha_max=5.0,
	                         dlogsigma_cs_N=np.log(1.01), sigma_cs_N_min=0.01, sigma_cs_N_max=10.0,
                                 dlogsigma_cs_Ca=np.log(1.01), sigma_cs_Ca_min=0.01, sigma_cs_Ca_max=10.0,
                                 dlogsigma_PF=np.log(1.01), sigma_PF_min=0.01, sigma_PF_max=10.0,               #GYH
                                 dloggamma=np.log(1.01), gamma_min=0.2, gamma_max=5.0,
                                 dalpha=0.1, alpha_min=0.0, alpha_max=0.1,                     
				 use_reference_prior_noe=True, use_reference_prior_H=False, use_reference_prior_Ha=False, use_reference_prior_N=False, use_reference_prior_Ca=False, use_reference_prior_PF=False, sample_ambiguous_distances=False, use_gaussian_reference_prior_noe=False, use_gaussian_reference_prior_H=False, use_gaussian_reference_prior_Ha=False, use_gaussian_reference_prior_N=False, use_gaussian_reference_prior_Ca=False, use_gaussian_reference_prior_PF=False)
  #sampler = PosteriorSampler(ensemble, use_reference_prior=True)
  sampler.sample(args.nsteps)  # number of steps
  print 'Processing trajectory...',
  sampler.traj.process()  # compute averages, etc.
  print '...Done.'

  print 'Writing results...',
  sampler.traj.write_results(os.path.join(args.outdir,'traj_lambda%2.2f.yaml'%args.lam))
  print '...Done.'

  # pickle the sampler object
  print 'Pickling the sampler object ...', 
  outfilename = 'sampler_lambda%2.2f.pkl'%args.lam
  print outfilename,
  fout = open(os.path.join(args.outdir, outfilename), 'wb')
  # Pickle dictionary using protocol 0.
  cPickle.dump(sampler, fout)
  print '...Done.' 


