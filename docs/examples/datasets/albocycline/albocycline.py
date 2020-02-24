#!/usr/bin/env python
# coding: utf-8

# <h1 style="align: center;font-size: 18pt;">Albocycline</h1>
# 
# <hr style="height:2.5px">
# to be completed...
# <!--
# Summary: In our previous work (DOI: [10.1002/jcc.23738](https://onlinelibrary-wiley-com.libproxy.temple.edu/doi/pdf/10.1002/jcc.23738")), we determine solution-state conformational populations of the 14-membered macrocycle cineromycin B, using a combination of previously published sparse Nuclear Magnetic Resonance (NMR) observables and replica exchange molecular dynamic/Quantum mehcanics (QM)-refined conformational ensembles. Cineromycin B is a 14-membered macrolide antibiotic that has become increasingly known for having activity against methicillin-resistant Staphylococcus Aureus (MRSA). In this example, we show how to calculate the consistency of computational modeling with experiment, and the relative importance of reference potentials and other model parameters. 
# -->
# <hr style="height:2.5px">

# In[5]:


import sys, os, glob, cPickle
import numpy as np
import biceps


# In[6]:


#########################################
# Lets' create input files for BICePs
############ Preparation ################
# Specify necessary argument values
path='NOE/*txt'        # precomputed distances for each state
states=100                    # number of states
indices='atom_indice_noe.txt'   # atom indices of each distance
exp_data='noe_distance.txt'  # experimental NOE data 
top='pdbs_guangfeng/0.pdb'    # topology file 
data_dir=path                 # directory of precomputed data 
dataFiles = 'noe_J'           # output directory of BICePs formated input file from this scripts
out_dir=dataFiles               

#p=Preparation('noe',states=states,indices=indices,exp_data=exp_data,top=top,data_dir=data_dir)  # the type of data needs to be specified {'noe', 'J', 'cs_H', etc}
#p.write(out_dir=out_dir)     # raw data will be converted to a BICePs readable format to the folder specified


# In[7]:


#########################################
# Let's create our ensemble of structures
############ Initialization #############    
# Specify necessary argument values
data = sort_data(dataFiles)   # sorting data in the folder and figure out what types of data are used
print data
print len(data),len(data[0])
#sys.exit(1)
energies_filename =  'albocycline_QMenergies.dat'
energies = loadtxt(energies_filename)
energies = loadtxt(energies_filename)*627.509  # convert from hartrees to kcal/mol
energies = energies/0.5959   # convert to reduced free energies F = f/kT
energies -= energies.min()  # set ground state to zero, just in case
outdir = 'results_ref_normal'
# Temporarily placing the number of steps here...
nsteps = 100 # number of steps of MCMC simulation

# Make a new directory if we have to
if not os.path.exists(outdir):
    os.mkdir(outdir)


# In[8]:


######################
# Main:
######################

lambda_values = [0.0,0.5,1.0]
res = list_res(data)
print res
ref=['uniform','exp']
uncern=[[0.05,20.0,np.log(1.02)],[0.05,5.0,np.log(1.02)]]
gamma = [0.05,2.0,np.log(1.02)]
#sys.exit()
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
           # if verbose:
           #     print File
            R=init_res('pdbs_guangfeng/%d.pdb'%i,lam,energies[i],ref[k],File,uncern[k], gamma)
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


# In[9]:


#########################################
# Let's do analysis using MBAR and plot figures
############ MBAR and Figures ###########
# Specify necessary argument values
get_ipython().run_line_magic('matplotlib', 'inline')
A = Analysis(100,dataFiles,outdir)    # number of states, input data files and output directory needs to be specified
A.plot()


# <h6 style="align: justify;font-size: 12pt"># <span style="color:red;">NOTE</span>: The following cell is for pretty notebook rendering</h6>

# In[1]:


from IPython.core.display import HTML
def css_styling():
    styles = open("../../theme.css", "r").read()
    return HTML(styles)
css_styling()

