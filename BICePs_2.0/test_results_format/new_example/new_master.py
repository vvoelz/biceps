### In BICePs 2.0, this script will play as a center role of BICePs calucltion.
### The users should specify all input files, type of reference potential
### they want to use (if other than default). --Yunhui 05/2018###

import sys, os, glob
sys.path.append('src') # source code path --Yunhui 04/2018
from BICePs import *
import datetime
#import sys, os, glob
#from Preparation import *
#from Restraint import *
#from PosteriorSampler import *
#from Analysis import *

#########################################
# Lets' create input files for BICePs
############ Preparation ################
# Specify necessary argument values
file_ext = 'h5'

path='cs_H/cs/H/*txt'
states=50
indices='cs_H/cs_indices_NH.txt'
exp_data='cs_H/chemical_shift_NH.txt'
top='cs_H/8690.pdb'
data_dir=path
out_dir='test_cs_H'
time=[]
#p=Preparation('cs_H',states=states,indices=indices,exp_data=exp_data,top=top,data_dir=data_dir)
#p.write(out_dir=out_dir)
print "start time: ", datetime.datetime.now()
time.append(datetime.datetime.now())
#a.append(datetime.datetime.now())
#print a
#sys.exit()
#########################################
# Let's create our ensemble of structures
############ Initialization #############
# Specify necessary argument values


dataFiles = 'test_cs_H'
data = sort_data(dataFiles)
energies_filename =  'energy.txt'
energies = loadtxt(energies_filename)
energies -= energies.min()  # set ground state to zero, just in case
outdir = 'results_ref_normal'
# Temporarily placing the number of steps here...
nsteps = 10000000 # 10000000
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
        print
        print '#### STRUCTURE %d ####'%i
	if verbose:
            print data[i]
        s = Restraint('8690.pdb',lam,energies[i],data = data[i])

        ensemble.append( s )



  ##########################################
  # Next, let's do some posterior sampling
  ########## Posterior Sampling ############

    sampler = PosteriorSampler(ensemble)

    sampler.sample(nsteps)  # number of steps
    print 'Processing trajectory...',
    sampler.traj.process()  # compute averages, etc.
    print '...Done.'

    print 'Writing results...',
    sampler.traj.write_results(os.path.join(outdir,'traj_lambda%2.2f.%s'%(lam,file_ext)))
    print '...Done.'

    # pickle the sampler object
    print 'Pickling the sampler object ...',
    outfilename = 'sampler_lambda%2.2f.pkl'%lam
    print outfilename,
    fout = open(os.path.join(outdir, outfilename), 'wb')
    # Pickle dictionary using protocol 0.
    cPickle.dump(sampler, fout)
    fout.close()
    print '...Done.'


    print "current time for lambda",j, ":", datetime.datetime.now()
    time.append(datetime.datetime.now())
#########################################
# Let's do analysis using MBAR and plot figures
############ MBAR and Figures ###########
# Specify necessary argument values
print time
sys.exit()


dataFiles = 'test_cs_H'
A = Analysis(50,dataFiles,'results_ref_normal')
A.plot()

