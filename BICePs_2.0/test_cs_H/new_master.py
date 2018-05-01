### In BICePs 2.0, this script will play as a center role of BICePs calucltion. The users should specify any input files, type of reference potential they want to use (if other than default). --Yunhui 04/2018###

import sys, os, glob

sys.path.append('../src') # source code path --Yunhui 04/2018


# import all necessary class for BICePs (Sturcture, Restraint, Sampling) --Yunhui 04/2018


from Structure import *
from PosteriorSampler import *

import cPickle  # to read/write serialized sampler classes

import argparse

# check all necessary input argument for BICePs --Yunhui 04/2018


parser = argparse.ArgumentParser()
parser.add_argument("lam", help="a lambda value between 0.0 and 1.0  denoting the Hamiltonian weight (E_data + lambda*E_QM)", type=float)	# this argument won't be necessary in the end of this upgrade, instead it will be one part of sampling and should be specified in Structure --Yunhui 04/2018

#parser.add_argument("energy", help="energy file to use") # This is not necessary --Yunhui 04/2018

parser.add_argument("outdir", help="the name of the output directory") # For now it's fine, but in the future I want this has a default set so users don't have to specify anything unless they want to --Yunhui 04/2018

parser.add_argument('--dataFiles','-f',nargs=None,required=True,
        metavar=None,help='Glob pattern for data')  # RMR
parser.add_argument("nsteps", help="Number of sampling steps", type=int) # We can keep it here --Yunhui 04/2018
parser.add_argument("--noref", help="Do not use reference potentials (default is to use them)",
                    action="store_true") # we need to check if this flag works well with our proposed modification. It should be fine to have it here as an "overall control" --Yunhui 04/2018

parser.add_argument("--lognormal", help="Use log-normal distance restraints (default is normal)",
                    action="store_true") # same as the last argument, need more check --Yunhui 04/2018

parser.add_argument("--verbose", help="use verbose output",
                    action="store_true") # will be useful for test but maybe not useful for actual calculation --Yunhui 04/2018

args = parser.parse_args()


print '=== Settings ==='
print 'lam', args.lam
#print 'enedgy', args.energy
print 'outdir', args.outdir
print 'nsteps', args.nsteps
print '--noref', args.noref
print '--lognormal', args.lognormal
print '--verbose', args.verbose


# RMR (this is a little specific...):{{{
if ',' in args.dataFiles:
    dir_list = (args.dataFiles).split(',')
    #print dir_list
    data = []
    for i in range(0,len(dir_list)):
        for j in sorted(glob.glob(dir_list[i])):
            data.append(j)
    print data
    print 'Number of data files: ',len(data)

else:
    data = sorted(glob.glob(args.dataFiles))

#}}}
#sys.exit(1)


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

if (1):
    verbose = False
#    nclusters = 50
    energies_filename =  'energy.txt'
    energies = loadtxt(energies_filename)
    if verbose:
        print 'energies.shape', energies.shape
    energies -= energies.min()  # set ground state to zero, just in case

# We will instantiate a number of Structure() objects to construct the ensemble
ensemble = []
#for i in range(energys.shape[0]):
for i in range(2):
    print
    print '#### STRUCTURE %d ####'%i

#    expdata = loadtxt('test_cs_H/ligand1_%d.cs_H'%i)
    #data = ['test_cs_H/ligand1_%d.cs_H'%i]

    s = Structure('8690.pdb', args.lam*energies[i],data = data)

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

    # add the structure to the ensemble
    ensemble.append( s )
#sys.exit()



  ##########################################
  # Next, let's do some posterior sampling


if (1):
    sampler = PosteriorSampler(ensemble, data=data,
            sample_ambiguous_distances=False)

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


