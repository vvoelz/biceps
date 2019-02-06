### In BICePs 2.0, this script will play as a center role of BICePs calucltion.
### The users should specify all input files, type of reference potential
### they want to use (if other than default). --Yunhui 05/2018###

import sys, os, glob
from numpy import *
sys.path.append('new_src')
from Preparation import *
from PosteriorSampler import *
from Analysis_new import *
from Restraint import *

#########################################
# Lets' create input files for BICePs
############ Preparation ################
# Specify necessary argument values
#path='J_coupling/*txt'
states=100
#indices='atom_indice_J.txt'
#exp_data='cs_H/chemical_shift_NH.txt'
top='cineromycinB_pdbs/0.fixed.pdb'
#data_dir=path
#dataFiles = 'test_cs_H'
dataFiles = 'noe_J'
out_dir=dataFiles

#p=Preparation('cs_H',states=states,indices=indices,exp_data=exp_data,top=top,data_dir=data_dir)
#p.write(out_dir=out_dir)

#p=Preparation('noe',states=15037,
#        exp_data='noe/noe.txt',
#        top='noe/Gens/Gens0.pdb',
#        data_dir='noe/micro_distances/dis_*.txt',
#        indices='noe/noe_indices.txt')
#p.write(out_dir=out_dir)

#########################################
# Let's create our ensemble of structures
############ Initialization #############
# Specify necessary argument values

data = sort_data(dataFiles)
print data
print len(data),len(data[0])
#sys.exit(1)
energies_filename =  'cineromycinB_QMenergies.dat'
energies = loadtxt(energies_filename)
energies = loadtxt(energies_filename)*627.509  # convert from hartrees to kcal/mol
energies = energies/0.5959   # convert to reduced free energies F = f/kT
np.savetxt('energy.dat',energies)
sys.exit()
energies -= energies.min()  # set ground state to zero, just in case
outdir = 'results_ref_normal'
# Temporarily placing the number of steps here...
nsteps = 10000000 # 10000000

# Make a new directory if we have to
if not os.path.exists(outdir):
    os.mkdir(outdir)

######################
# Main:
######################

lambda_values = [0.0,0.5,1.0]
for j in lambda_values:
    verbose = True#False
    lam = j
    # We will instantiate a number of Structure() objects to construct the ensemble
    ensemble = []
    for i in range(energies.shape[0]):
        print '\n#### STRUCTURE %d ####'%i
        ensemble.append([])
        for k in range(len(data[0])):
            File = data[i][k]
            if verbose:
                print File

            # Call on the Restraint that corresponds to File
            if File.endswith('cs_H'):
                R = Restraint_cs_H('8690.pdb',ref='exp')
                R.prep_observable(lam=lam, free_energy=energies[i],
                        filename=File)


            elif File.endswith('cs_CA'):
                R = Restraint_cs_Ca('8690.pdb',ref='gaussian')
                R.prep_observable(lam=lam, free_energy=energies[i],
                        filename=File)

            elif File.endswith('cs_Ha'):
                R = Restraint_cs_Ha('8690.pdb',ref='exp')
                R.prep_observable(lam=lam, free_energy=energies[i],
                        filename=File)

            elif File.endswith('cs_N'):
                R = Restraint_cs_N('8690.pdb',ref='gaussian')
                R.prep_observable(lam=lam, free_energy=energies[i],
                        filename=File)

            elif File.endswith('J'):
                R = Restraint_J('cineromycinB_pdbs/%d.fixed.pdb'%i,ref='uniform', dlogsigma=np.log(1.02), sigma_min=0.05, sigma_max=20.0)  # good ref
                R.prep_observable(lam=lam, free_energy=energies[i],
                        filename=File)

            elif File.endswith('noe'):
                R = Restraint_noe('cineromycinB_pdbs/%d.fixed.pdb'%i,ref='exp',dlogsigma=np.log(1.02), sigma_min=0.05, sigma_max=5.0)   # good ref
                R.prep_observable(lam=lam, free_energy=energies[i],
                        filename=File, dloggamma=np.log(1.01),gamma_min=0.5,gamma_max=2.0)

            elif File.endswith('pf'):
                R = Restraint_pf('8690.pdb',ref='gaussian')
                R.prep_observable(lam=lam, free_energy=energies[i],
                        filename=File)

            ensemble[-1].append(R)
    print ensemble

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


#sys.exit(1)

#########################################
# Let's do analysis using MBAR and plot figures
############ MBAR and Figures ###########
# Specify necessary argument values

A = Analysis(100,dataFiles,outdir)
A.plot()

