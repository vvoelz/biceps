# import source code
import sys, os, glob
from numpy import *
sys.path.append('biceps')
from Preparation import *
from PosteriorSampler import *
from Analysis import *
from Restraint import *
from init_res import *

############ Initialization #############
# Specify necessary argument values

# REQUIRED: specify number of states
#states=100
states = YOUR STATES NUMBER

# REQUIRED: specify directory of input data (BICePs readable format)
#dataFiles = 'noe_J'
dataFiles = 'YOUR INPUT FILES'

# REQUIRED: sort data and figure out what experimental restraints are included for each state
data = sort_data(dataFiles)

# REQUIRED: energy file name of each state (computational prior distribution)

#energies_filename = 'energy.dat'
energies_filename = 'YOUR ENERGY FILE'
energies = loadtxt(energies_filename)
energies -= energies.min()  # set ground state to zero, just in case

# REQUIRED: specify outcome directory of BICePs sampling
outdir = 'results_ref_normal'
# Make a new directory if we have to
if not os.path.exists(outdir):
    os.mkdir(outdir)

# REQUIRED: number of MCMC steps for each lambda
#nsteps = 1000000
nsteps = YOUR NUMBER OF STEPS FOR MC SAMPLING

# REQUIRED: specify how many lambdas to sample (more lambdas will provide higher accuracy but slower the whole process, lambda=0.0 and 1.0 are necessary)
lambda_values = [0.0,0.5,1.0]

# OPTIONAL but RECOMMENDED: print experimental restraints included (a chance for double check)
res = list_res(data)
print res

# OPTIONAL: specify reference potential to use for each experimental observable
# will be in the same order as the printed observables from (print res)
#ref=['uniform','exp']
ref = ['YOUR SELECTED REFERENCE POTENTIAL FOR EACH OBSERVABLE']

# OPTIONAL: specify nuisance parameters for each experimnetal observable
# will be in the same order as the printed observables from (print res)
# only specify if you want to narrow down the default range
#uncern=[[0.05,20.0,1.02],[0.05,5.0,1.02]]
#gamma = [0.2,5.0,1.01]
uncern = [YOUR NUISANCE PARAMETERS RANGE]
gamma = [YOUR GAMMA PARAMETERS RANGE]


######################
# Main:
######################

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
            #R=init_res('top/%d.fixed.pdb'%i,lam,energies[i],ref[k],File,uncern[k],gamma)
            R=init_res('YOUR TOPOLOGY FILE',lam,energies[i],ref[k],File,uncern[k],gamma)
            ensemble[-1].append(R)
        #print ensemble

    ##########################################
    # Next, let's do posterior MCMC sampling
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
# Let's do analysis using MBAR algorithm and plot figures
############ MBAR and Figures ###########
# Specify necessary argument values

#A = Analysis(100,dataFiles,outdir)
A = Analysis(states,dataFiles,outdir)
A.plot()

