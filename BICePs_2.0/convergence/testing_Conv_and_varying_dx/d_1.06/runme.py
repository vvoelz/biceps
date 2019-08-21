### In BICePs 2.0, this script will play as a center role of BICePs calucltion.
### The users should specify all input files, type of reference potential
### they want to use (if other than default). --Yunhui 05/2018###

import sys, os, glob
from numpy import *
sys.path.append('biceps')
from Preparation import *
from PosteriorSampler import *
from Analysis import *
from Restraint import *
from init_res import *
#########################################
# Lets' create input files for BICePs
############ Preparation ################
# Specify necessary argument values
states=100
top='../cineromycinB_pdbs/0.fixed.pdb'
dataFiles = '../noe_J'
out_dir=dataFiles


#########################################
# Let's create our ensemble of structures
############ Initialization #############
# Specify necessary argument values

data = sort_data(dataFiles)
energies_filename =  '../cineromycinB_QMenergies.dat'
energies = loadtxt(energies_filename)
energies = loadtxt(energies_filename)*627.509  # convert from hartrees to kcal/mol
energies = energies/0.5959   # convert to reduced free energies F = f/kT
energies -= energies.min()  # set ground state to zero, just in case
outdir = 'results_ref_normal'
nsteps = 10000000

# Make a new directory if we have to
if not os.path.exists(outdir):
    os.mkdir(outdir)

uncern=[[0.05,20.0,1.06],[0.05,5.0,1.06]]
gamma = [0.2,5.0,1.06] 

ref=['uniform','exp']

######################
# Main:
######################

lambda_values = [0.0,0.5,1.0]

for j in lambda_values:
    print 'lambda', j
    verbose = False #False
    lam = j
    # We will instantiate a number of Restraint() objects to construct the ensemble
    # experimental data and pre-computed model data are compiled for each state
    ensemble = []
    for i in range(energies.shape[0]):   # number of states
        if verbose:
            print '\n#### STRUCTURE %d ####'%i
        ensemble.append([])
        for k in range(len(data[0])):   # number of experimental observables
            File = data[i][k]
            if verbose:
                print File
            R=init_res('../cineromycinB_pdbs/0.fixed.pdb',lam,energies[i],File,ref[k],uncern[k],gamma)
            ensemble[-1].append(R)

#    print ensemble

    ##########################################
    # Next, let's do some posterior sampling
    ########## Posterior Sampling ############

    sampler = PosteriorSampler(ensemble)
    sampler.compile_nuisance_parameters()

    sampler.sample(nsteps)  # number of steps

    print 'Processing trajectory...',

    sampler.traj.process()  # compute averages, etc.
    print '...Done.'

    print 'Writing results...',
    sampler.traj.write_results(os.path.join(outdir,'traj_lambda%2.2f.npz'%lam))
    print '...Done.'
    sampler.traj.read_results(os.path.join(outdir,'traj_lambda%2.2f.npz'%lam))

    print 'Pickling the sampler object ...',
    outfilename = 'sampler_lambda%2.2f.pkl'%lam
    print outfilename,
    fout = open(os.path.join(outdir, outfilename), 'wb')
    # Pickle dictionary using protocol 0.
    cPickle.dump(sampler, fout)
    fout.close()
    print '...Done.'



#########################################
# Let's do analysis using MBAR and plot figures
############ MBAR and Figures ###########
# Specify necessary argument values

A = Analysis(100,dataFiles,outdir)
A.plot()

