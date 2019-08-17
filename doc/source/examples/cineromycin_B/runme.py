### In BICePs 2.0, this script will play as a center role of BICePs calucltion.
### The users should specify all input files, type of reference potential
### they want to use (if other than default). --Yunhui 05/2018###

import sys, os, glob, cPickle
import numpy as np
import biceps


#########################################
# Lets' create input files for BICePs
############ Preparation ################
# Specify necessary argument values
path='NOE/*txt'                # precomputed distances for each state
states=100                     # number of states
indices='atom_indice_noe.txt'  # atom indices of each distance
exp_data='noe_distance.txt'    # experimental NOE data
top='cineromycinB_pdbs/0.fixed.pdb'    # topology file
dataFiles = 'noe_J'            # output directory of BICePs formated input file from this scripts
out_dir=dataFiles

#p=Preparation('noe',states=states,indices=indices,
#    exp_data=exp_data,top=top,data_dir=data_dir)  # the type of data needs to be specified {'noe', 'J', 'cs_H', etc}
#p.write(out_dir=out_dir)      # raw data will be converted to a BICePs readable format to the folder specified

#########################################
# Let's create our ensemble of structures
############ Initialization #############
# Specify necessary argument values
data = biceps.sort_data(dataFiles)   # sorting data in the folder and figure out what types of data are used
print data
print len(data),len(data[0])

energies_filename =  'cineromycinB_QMenergies.dat'
energies = np.loadtxt(energies_filename)*627.509  # convert from hartrees to kcal/mol
energies = energies/0.5959   # convert to reduced free energies F = f/kT
energies -= energies.min()  # set ground state to zero, just in case
outdir = 'RR_results_ref_normal'

# Make a new directory if we have to
if not os.path.exists(outdir):
    os.mkdir(outdir)

nsteps = 1000000 # number of steps of MCMC simulation

#lambda_values = [0.0,0.5,1.0]
#lambda_values = [0.0,1.0]
lambda_values = [0.0]

res = biceps.list_res(data)

ref=['uniform','exp']
uncern=[[0.05,20.0,1.02],[0.05,5.0,1.02]]
gamma = [0.2,5.0,1.01]


######################
# Main:
######################

for j in lambda_values:
    verbose = False # True
    lam = j
    # We will instantiate a number of Structure() objects to construct the ensemble
    ensemble = []
    for i in range(energies.shape[0]):
        if verbose:
            print '\n#### STRUCTURE %d ####'%i
        ensemble.append([])
        for k in range(len(data[0])):
            File = data[i][k]
            if verbose:
                print File
            R = biceps.init_res('cineromycinB_pdbs/0.fixed.pdb',
                              lam,energies[i],File,ref[k],uncern[k],gamma)
            ensemble[-1].append(R)
    print ensemble

    ##########################################
    # Next, let's do some posterior sampling
    ########## Posterior Sampling ############

    sampler = biceps.PosteriorSampler(
            ensemble=ensemble,freq_write_traj=1.,
            freq_print=1., freq_save_traj=1.,)

    sampler.compile_nuisance_parameters()

    sampler.sample(nsteps)  # number of steps

    print 'Processing trajectory...',
    print 'Writing results...',
    sampler.traj.process_results(os.path.join(outdir,'traj_lambda%2.2f.npz'%lam))  # compute averages, etc.
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

exit(1)
#########################################
# Let's do analysis using MBAR and plot figures
############ MBAR and Figures ###########
# Specify necessary argument values

A = Analysis(100,dataFiles,outdir)
A.plot()

