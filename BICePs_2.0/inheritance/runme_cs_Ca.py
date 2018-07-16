### In BICePs 2.0, this script will play as a center role of BICePs calucltion.
### The users should specify all input files, type of reference potential
### they want to use (if other than default). --Yunhui 05/2018###

import sys, os, glob
sys.path.append('src')
from Preparation import *
from numpy import *
from PosteriorSampler import *
from Analysis import *
from restraint_cs_Ca import *


#########################################
# Lets' create input files for BICePs
############ Preparation ################
# Specify necessary argument values
path='cs_Ca/cs/Ca/*txt'
states=50
indices='cs_Ca/cs_indices_Ca.txt'
exp_data='cs_Ca/chemical_shift_Ca.txt'
top='cs_Ca/8690.pdb'
data_dir=path
dataFiles = 'test_cs_Ca'
out_dir=dataFiles

# Going to skip this portion of the runme.py script

#p=Preparation('cs_Ca',states=states,indices=indices,exp_data=exp_data,top=top,data_dir=data_dir)
#p.write(out_dir=out_dir)


#########################################
# Let's create our ensemble of structures
############ Initialization #############
# Specify necessary argument values

data = sort_data(dataFiles)
energies_filename =  'energy.txt'
energies = loadtxt(energies_filename)
energies -= energies.min()  # set ground state to zero, just in case
outdir = 'results_ref_normal_cs_Ca'
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

lambda_values = [0.0,1.0]
for j in lambda_values:
    verbose = False
    lam = j
    # We will instantiate a number of Structure() objects to construct the ensemble
    ensemble = []
    for i in range(energies.shape[0]):
        print '\n#### STRUCTURE %d ####'%i
	if verbose:
            print data[i]
        s = Restraint_cs_Ca('8690.pdb',lam,
                energies[i],data=data[i])
        s.load_data(str(data[i][0]))#,verbose=True)
        ensemble.append( s )
    print ensemble
    #sys.exit(1)


  ##########################################
  # Next, let's do some posterior sampling
  ########## Posterior Sampling ############


    sampler = PosteriorSampler(ensemble)
    sampler.sample(nsteps)  # number of steps
    print 'Processing trajectory...',

    sampler.traj.process()  # compute averages, etc.
    print '...Done.'

    print 'Writing results...',
    sampler.traj.write_results(os.path.join(outdir,'traj_lambda%2.2f.npz'%lam))
    print '...Done.'
    sampler.traj.read_results(os.path.join(outdir,'traj_lambda%2.2f.npz'%lam))

    # pickle the sampler object
    print 'Pickling the sampler object ...',
    outfilename = 'sampler_lambda%2.2f.pkl'%lam
    print outfilename,
    fout = open(os.path.join(outdir, outfilename), 'wb')
    # Pickle dictionary using protocol 0.
    cPickle.dump(sampler, fout)
    fout.close()
    print '...Done.'



sys.exit(1)

#########################################
# Let's do analysis using MBAR and plot figures
############ MBAR and Figures ###########
# Specify necessary argument values

dataFiles = 'test_cs_Ca'
A = Analysis(50,dataFiles,outdir)
A.plot()

