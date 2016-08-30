import sys, os, glob

sys.path.append('./')
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
expdata_filename_noe = 'exp_distances.dat' 
#expdata_filename_J = ''
#expdata_filename_cs = 'trploop2a.chemicalshift'
#expdata_filename_PF = 'trploop2a.PF'		#GYH
# model energies
if (1):
    nclusters = 72
    energies_filename = 'reduced_free_energies_eps0.000.dat'
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

for i in range(nclusters+1):

    print
    print '#### STRUCTURE %d ####'%i

    # no information from QM --> lam = 0.0
    # QM + exp               --> lam = 1.0
    ## s = Structure('gens-pdb-kcenters-dih-1.8/Gen%d.pdb'%i, args.lam*energies[i], expdata_filename, use_log_normal_distances=False)
    model_distances = loadtxt('mean_distances/mean_distances_state%d.dat'%i)*10.0 # convert to A
#    model_chemicalshift = loadtxt('chemical_shift/cs/cs_%d.txt'%i)
#    s = Structure('Gens/Gens%d.pdb'%i, args.lam*energies[i], expdata_filename_noe, expdata_filename_J, expdata_filename_cs, use_log_normal_distances=False, dloggamma=np.log(1.01), gamma_min=0.2, gamma_max=5.0)
    s = Structure('Gens/Gens%d.pdb'%i, args.lam*energies[i], expdata_filename_noe=expdata_filename_noe, expdata_filename_cs=expdata_filename_cs, expdata_filename_PF=expdata_filename_PF,  use_log_normal_distances=False, dloggamma=np.log(1.01), gamma_min=0.2, gamma_max=5.0)	#GYH
#    s = Structure('Gens/Gens%d.pdb'%i, args.lam*energies[i], expdata_filename_noe=expdata_filename_noe, use_log_normal_distances=False, dloggamma=np.log(1.01), gamma_min=0.2, gamma_max=5.0)

    # NOTE: Upon instantiation, each Structure() object computes the distances from the given PDB.
    #       However, our clusters represent averaged conformation states, and so we   
    #       need to replace these values with our own r^-6-averaged, precomputed ones

    # replace PDB distances with r^-6-averaged distances
    print 'len(s.distance_restraints)', len(s.distance_restraints)
    for j in range(len(s.distance_restraints)):
        print s.distance_restraints[j].i, s.distance_restraints[j].j, model_distances[j]
        s.distance_restraints[j].model_distance = model_distances[j]

    # replace chemicalshift  with  precomputed ones
    print 'len(s.chemicalshift_restraints)', len(s.chemicalshift_restraints)
    for j in range(len(s.chemicalshift_restraints)):
        print s.chemicalshift_restraints[j].i, model_chemicalshift[j]
        s.chemicalshift_restraints[j].model_chemicalshift = model_chemicalshift[j]

    # replace protection_factor with precomputed ones				#GYH
    print 'len(s.protectionfactor_restraints)', len(s.protectionfactor_restraints)
    for j in range(len(s.protectionfactor_restraints)):
        print s.protectionfactor_restraints[j].i, model_protectionfactor[j]
        s.protectionfactor_restraints[j].model_protectionfactor = model_protectionfactor[j]


    # update the chemicalshift sse's!
    s.compute_sse_chemicalshift()

    # update the distance sse's!			
    s.compute_sse_distances()

    # update the protectionfactor sse's!		#GYH
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
                                 dlogsigma_cs=np.log(1.02), sigma_cs_min=0.01, sigma_cs_max=20.0,
				 dlogsigma_PF=np.log(1.02), sigma_PF_min=0.01, sigma_PF_max=20.0		#GYH
				 dloggamma=np.log(1.01), gamma_min=0.2, gamma_max=5.0,
                                 use_reference_prior=not(args.noref), sample_ambiguous_distances=False)
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


