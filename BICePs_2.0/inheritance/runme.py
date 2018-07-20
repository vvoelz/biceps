### In BICePs 2.0, this script will play as a center role of BICePs calucltion.
### The users should specify all input files, type of reference potential
### they want to use (if other than default). --Yunhui 05/2018###

import sys, os, glob
from numpy import *
sys.path.append('src')
from Preparation import *
from PosteriorSampler import *
from Analysis import *
from Restraint import *

#########################################
# Lets' create input files for BICePs
############ Preparation ################
# Specify necessary argument values
path='cs_H/cs/H/*txt'
states=50
indices='cs_H/cs_indices_NH.txt'
exp_data='cs_H/chemical_shift_NH.txt'
top='cs_H/8690.pdb'
data_dir=path
dataFiles = 'test_cs_H'
out_dir=dataFiles

p=Preparation('cs_H',states=states,indices=indices,exp_data=exp_data,top=top,data_dir=data_dir)
p.write(out_dir=out_dir)

#########################################
# Let's create our ensemble of structures
############ Initialization #############
# Specify necessary argument values

data = sort_data(dataFiles)
energies_filename =  'energy.txt'
energies = loadtxt(energies_filename)
energies -= energies.min()  # set ground state to zero, just in case
outdir = 'results_ref_normal_cs_H'
# Temporarily placing the number of steps here...
nsteps = 1000 # 10000000

# Make a new directory if we have to
if not os.path.exists(outdir):
    os.mkdir(outdir)

######################
# Main:
######################

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
        if 'cs_H' in data[i][0].split('.')[-1]:
            R = Restraint_cs_H('8690.pdb')
            R.prep_observable(lam=lam, free_energy=energies[i],
                    filename=data[i][0])
            print 'Loaded ',data[i][0]

        elif 'cs_Ca' in data[i][0].split('.')[-1]:
            R = Restraint_cs_Ca('8690.pdb')
            R.prep_observable(lam=lam, free_energy=energies[i],
                    filename=data[i][0])
            print 'Loaded ',data[i][0]

        elif 'cs_Ha' in data[i][0].split('.')[-1]:
            R = Restraint_cs_Ha('8690.pdb')
            R.prep_observable(lam=lam, free_energy=energies[i],
                    filename=data[i][0])
            print 'Loaded ',data[i][0]

        elif 'cs_N' in data[i][0].split('.')[-1]:
            R = Restraint_cs_N('8690.pdb')
            R.prep_observable(lam=lam, free_energy=energies[i],
                    filename=data[i][0])
            print 'Loaded ',data[i][0]

        elif 'J' in data[i][0].split('.')[-1]:
            R = Restraint_J('8690.pdb')
            R.prep_observable(lam=lam, free_energy=energies[i],
                    filename=data[i][0])
            print 'Loaded ',data[i][0]

        elif 'noe' in data[i][0].split('.')[-1]:
            R = Restraint_noe('8690.pdb')
            R.prep_observable(lam=lam, free_energy=energies[i],
                    filename=data[i][0])
            print 'Loaded ',data[i][0]

        elif 'pf' in data[i][0].split('.')[-1]:
            R = Restraint_pf('8690.pdb')
            R.prep_observable(lam=lam, free_energy=energies[i],
                    filename=data[i][0])
            print 'Loaded ',data[i][0]

        ensemble.append(R)
        #sys.exit(1)
    print ensemble


  ##########################################
  # Next, let's do some posterior sampling
  ########## Posterior Sampling ############

    sampler = PosteriorSampler(ensemble)
    sampler.sample(nsteps)  # number of steps
    print 'Processing trajectory...',
#    sys.exit(1)
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

dataFiles = 'test_cs_H'
A = Analysis(50,dataFiles,outdir)
A.plot()

