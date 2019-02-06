import sys, os, glob

sys.path.append('../../src')

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

# exp data
#expdata_filename = 'cilengitide_NOE_Data_VAV_USETHIS.yaml'
expdata_filename = 'cilengitide_NOE_Data_VAV_USETHIS_plainavg.yaml'


# model energies
if (1):
    nclusters = 100
    energies_filename = 'updated/reduced_free_energies.dat'
    energies = loadtxt(energies_filename)
    print 'energies.shape', energies.shape
    energies -= energies.min()  # set ground state to zero, just in case


if (1):
    # model distances 
    model_distances = loadtxt('updated_VAV/minus6distances-for-%d.dat'%nclusters)**(-1./6.)
    print 'model_distances.shape', model_distances.shape
    print 'model_distances', model_distances

    # There are 41 distances, but these are averaged over equivalent pairs!
    model_distance_column_index = []  # a lookup table for which column to use in model_distances
    fin = open('rgdf_nmev_NOE_Data_VAV_USETHIS.dat','r')
    lines = fin.readlines()
    for line in lines:
        if line[0] != '#':
            fields = line.split()
            if len(fields) > 7:
                model_distance_column_index.append( int(fields[7])-1 )
    print 'len(model_distance_column_index)', len(model_distance_column_index)
    print model_distance_column_index

############

ensemble = []
for i in range(nclusters):

    print
    print '#### STRUCTURE %d ####'%i

    # no information from QM --> lam = 0.0
    # QM + exp               --> lam = 1.0
    ## s = Structure('gens-pdb-kcenters-dih-1.8/Gen%d.pdb'%i, args.lam*energies[i], expdata_filename, use_log_normal_distances=False)
    s = Structure('updated/Gens/Gen%d.pdb'%i, args.lam*energies[i], expdata_filename, use_log_normal_distances=False)
    # upon instantiation, the distances get computed from the structure, but these are clusters so
    # we need to replace these values with our own r^-6-averaged, precomputed ones

    print 'len(s.distance_restraints)', len(s.distance_restraints)
    for j in range(len(s.distance_restraints)):
        #print 'i', i, 'j', j, 'model_distance_column_index[j]', model_distance_column_index[j]
        print s.distance_restraints[j].i, s.distance_restraints[j].j, model_distance_column_index[j]
        s.distance_restraints[j].model_distance = model_distances[i,model_distance_column_index[j]]
    ensemble.append( s )

# Print out the agreement for model 53 (highest-pop)
for drest in ensemble[53].distance_restraints:
    print 'state 53 d[ %d - %d ] ='%(drest.i, drest.j), drest.model_distance, 'd_exp =', drest.exp_distance

model_distances = [drest.model_distance for drest in ensemble[53].distance_restraints]
exp_distances = [drest.exp_distance for drest in ensemble[53].distance_restraints]

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
  sampler = PosteriorSampler(ensemble, dlogsigma_noe=np.log(1.02), sigma_noe_min=0.05, sigma_noe_max=5.0,
                                 dlogsigma_J=np.log(1.02), sigma_J_min=0.05, sigma_J_max=20.0,
                                 dloggamma=np.log(1.01), gamma_min=0.5, gamma_max=2.0,
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


