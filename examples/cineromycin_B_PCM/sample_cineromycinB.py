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

expdata_filename = 'cineromycinB_expdata_VAV.yaml'
#expdata_filename = 'cineromycinB_expdata.yaml'  <-- doesn't contain the right Karplus specs 
energies_filename = 'cineromycinB_QMenergies_PCM.dat'
energies = loadtxt(energies_filename)*627.509  # convert from hartrees to kcal/mol
energies = energies/0.5959   # convert to reduced free energies F = f/kT
energies -= energies.min()

ensemble = []
for i in range(100):

    print
    print '#### STRUCTURE %d ####'%i

    # no information from QM --> lam = 0.0
    # QM + exp               --> lam = 1.0
    ensemble.append( Structure('cineromycinB_pdbs/%d.fixed.pdb'%i, args.lam*energies[i], expdata_filename, use_log_normal_distances=False) )

    
if (0):
  # collect distance distributions for each distance and plot them for selected states
  selected_states = [38] # the "native" state 38
  selected_states.append( 87 )  # state that fits restraints well but is high QM energy
  nselected = len(selected_states)
  s = ensemble[0]
  ndistances = len(s.distance_restraints)
  all_distances = []
  distance_distributions = [[] for i in range(ndistances)]
  for s in ensemble:
    for i in range(ndistances):
        distance_distributions[i].append( s.distance_restraints[i].model_distance )
        all_distances.append( s.distance_restraints[i].model_distance )

  plt.figure()
  for k in range(nselected): 
      plt.subplot(np.ceil(nselected**0.5), np.ceil(nselected**0.5), k+1)
      plt.plot( [0,ndistances+1],[2.5,2.5], 'r-')
      plt.hold(True)
      for i in range(ndistances):
          weight = ensemble[selected_states[k]].distance_restraints[i].weight
          print i, ensemble[selected_states[k]].distance_restraints[i].model_distance, weight
          plt.plot(i, ensemble[selected_states[k]].distance_restraints[i].model_distance,'ko', markersize=10.0*weight)
          plt.hold(True)
  plt.show()

# for debugging
#sys.exit(1)


if (0):
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

#sys.exit(1)

##########################################
# Next, let's do some posterior sampling


if (1):
  #sampler = PosteriorSampler(ensemble, use_reference_prior=True, sample_ambiguous_distances=False)
  sampler = PosteriorSampler(ensemble, dlogsigma_noe=np.log(1.02), sigma_noe_min=0.05, sigma_noe_max=5.0,
                                 dlogsigma_J=np.log(1.02), sigma_J_min=0.05, sigma_J_max=20.0,
                                 dloggamma=np.log(1.01), gamma_min=0.5, gamma_max=2.0,
                                 use_reference_prior=not(args.noref), sample_ambiguous_distances=False)

  #sampler = PosteriorSampler(ensemble, use_reference_prior=True)
  sampler.sample(args.nsteps)  # number of steps
  sampler.traj.process()  # compute averages, etc.
  sampler.traj.write_results(os.path.join(args.outdir,'traj_lambda%2.2f.yaml'%args.lam))
  # pickle the sampler object
  fout = open(os.path.join(args.outdir,'sampler_lambda%2.2f.pkl'%args.lam), 'wb')
  # Pickle dictionary using protocol 0.
  cPickle.dump(sampler, fout)


