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
path='NOE/rminus*txt'
states=100
indices='atom_indice_noe.txt'
exp_data='noe_distance.txt'
top='pdb/state_00.pdb'
data_dir=path
dataFiles = 'test_NOE'
out_dir=dataFiles

#p=Preparation('noe',states=states,
#        indices=indices,exp_data=exp_data,
#        top=top,data_dir=data_dir)
#p.write(out_dir=out_dir)

#########################################
# Let's create our ensemble of structures
############ Initialization #############
# Specify necessary argument values

data = sort_data(dataFiles)
print data
print len(data),len(data[0])
energies_filename =  'energy.txt'
energies = loadtxt(energies_filename)
energies -= energies.min()  # set ground state to zero, just in case
outdir = 'results_ref_normal_TEST_NOE'
nsteps = 1000000 # 10000000

# Make a new directory if we have to
if not os.path.exists(outdir):
    os.mkdir(outdir)

######################
# Main:
######################
lambda_values = [0.0,1.0]
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
                R = Restraint_cs_H('pdb/state_%02d.pdb'%i,ref='exp')
                R.prep_observable(lam=lam, free_energy=energies[i],
                        filename=File)
                # Change the experimental Uncertainty after prepping observable
                R.exp_uncertainty(dlogsigma=np.log(1.02),sigma_min=0.05,
                        sigma_max=20.0)

            elif File.endswith('cs_CA'):
                R = Restraint_cs_Ca('pdb/state_%02d.pdb'%i,ref='gaussian')
                R.prep_observable(lam=lam, free_energy=energies[i],
                        filename=File)

            elif File.endswith('cs_Ha'):
                R = Restraint_cs_Ha('pdb/state_%02d.pdb'%i,ref='exp')
                R.prep_observable(lam=lam, free_energy=energies[i],
                        filename=File)
                # Change the experimental Uncertainty after prepping observable
                R.exp_uncertainty(dlogsigma=np.log(1.02),sigma_min=10,
                        sigma_max=20.0)

            elif File.endswith('cs_N'):
                R = Restraint_cs_N('pdb/state_%02d.pdb'%i,ref='gaussian')
                R.prep_observable(lam=lam, free_energy=energies[i],
                        filename=File)

            elif File.endswith('J'):
                R = Restraint_J('pdb/state_%02d.pdb'%i,ref='uniform')  # good ref
                R.prep_observable(lam=lam, free_energy=energies[i],
                        filename=File)
                R.exp_uncertainty(dlogsigma=np.log(1.02),sigma_min=5,
                        sigma_max=20.0)

            elif File.endswith('noe'):
                R = Restraint_noe('pdb/state_%02d.pdb'%i,ref='exp')   # good ref
                R.prep_observable(lam=lam, free_energy=energies[i],
                        filename=File)
                # Change the experimental Uncertainty after prepping observable
                R.exp_uncertainty(dlogsigma=np.log(1.01),sigma_min=0.2,
                        sigma_max=10.0)

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
    sampler.traj.write_results(os.path.join(outdir,'RR_traj_lambda%2.2f.npz'%lam))
    print '...Done.'
    sampler.traj.read_results(os.path.join(outdir,'RR_traj_lambda%2.2f.npz'%lam))

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

A = Analysis(50,dataFiles,outdir)
A.plot()

