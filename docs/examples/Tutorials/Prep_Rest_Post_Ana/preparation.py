#!/usr/bin/env python
# coding: utf-8

# <h1 style="align: center;font-size: 18pt;">Preparation</h1>
# 
# <hr style="height:2.5px">
# 
# This tutorial shows the user how to properly use methods in the `Preparation class` to prepare input files for the next step, which is constructing the ensemble via the `BICePs` `Restraint class`. The data from this tutorial can be found from this work (DOI: [10.1002/jcc.23738](https://onlinelibrary.wiley.com/doi/full/10.1002/jcc.23738)).
# 
# <hr style="height:2.5px">

# In[1]:


import sys, os
import biceps


# In this tutorial, we have two experimental observables: (1) [J couplings](https://en.wikipedia.org/wiki/J-coupling) for small molecules and (2) [NMR nuclear Overhauser effect (NOE)](https://en.wikipedia.org/wiki/Nuclear_Overhauser_effect) (pairwise distances).
# First we need to perform conformational clustering on our MD simulations data. In this case, 100 metastable states are clustered. Now we need to prepare prior knowledge we learned from computational simulations. Normally, we used potential energy for each metastable state. In the [original work](https://onlinelibrary.wiley.com/doi/full/10.1002/jcc.23738), Zhou et al did Quantum Mechanical (QM) calculations to refine each state and B3LYP energy was used as priors in BICePs calculation. Instructions of QM calculations are beyond the scope of this tutorial. Alternatively, we can estimate the potential energy using U = -ln(P) where P is the normalized population for each state. We also have built-in functions ([toolbox](https://biceps.readthedocs.io/en/latest/api.html#toolbox)) to conduct this conversion. You can find tutorials using functions in `toolbox.py` [here](https://biceps.readthedocs.io/en/latest/tutorials/Tools/toolbox.html).   
# 

# Next, we need to compute pairwise distances and J coupling constants for each clustered state. 
# To compute pairwise distances, we recommend to use [MDTraj](http://mdtraj.org) which is free to download. 
# To compare simulated conformational ensembles to experimental NOE measurements, we normally computed $<r^{-6}>^{-1/6}$. For convenience in this tutorial, we assume the cluster center of each state is representative enough and simply compute pairwise distances for the cluster center conformation. In practice, we still recommend users to compute ensemble-averaged distance.

# In[2]:


import mdtraj as md
import numpy as np

data_dir = "../../datasets/cineromycin_B/"
# atom indices of pairwise distances
ind=np.loadtxt(data_dir+'atom_indice_noe.txt')
print("indices", ind)

# make a new folder of computed distances for later 
os.system(data_dir+'mkdir NOE')

# compute pairwise distances using MDTraj
for i in range(100):    # 100 clustered states
    #print('state', i)
    t = md.load(data_dir+'cineromycinB_pdbs/%d.fixed.pdb'%i)
    d=md.compute_distances(t,ind)*10.     # convert nm to Å 
    np.savetxt(data_dir+'NOE/%d.txt'%i,d)
print("Done!")


# Next, we need to convert computed distance to BICePs readable format. 

# In[3]:


#########################################
# Lets' create input files for BICePs
############ Preparation ################
# Specify necessary argument values

# REQUIRED: raw data of pre-comuted chemical shifts
path = data_dir+'NOE/*txt'

# REQUIRED: number of states
states = 100

# REQUIRED: atom indices of pairwise distances
indices = data_dir+'atom_indice_noe.txt'

# REQUIRED: experimental data
exp_data = data_dir+'noe_distance.txt'

# REQUIRED: topology file (as it only supports topology information, so it doesn't matter which state is used)
top = data_dir+'cineromycinB_pdbs/0.fixed.pdb'

# OPTIONAL: output directory of generated files
out_dir = data_dir+'noe_J'

p = biceps.Preparation('noe',states=states,indices=indices,exp_data=exp_data,top=top,data_dir=path)   # 'noe' scheme is selected
p.write(out_dir=out_dir)
# This will convert pairwise distances files for each state to a BICePs readable format and saved the new files in "noe_J" folder.


# Now, let's take a look at what's inside the newly generated files.

# In[4]:


fin = open(data_dir+'noe_J/0.noe','r')
text = fin.read()
fin.close()
print(text)


# Now let's move on to J couplings. Model predictions of coupling constants from dihedral angles θ were obtained from Karplus relations chosen depending on the relevant stereochemistry. 

# In[6]:


# atom indices of J coupling constants
ind=np.load(data_dir+'ind.npy')
print('index', ind)

# Karplus relations for each dihedral angles 
karplus_key=np.loadtxt(data_dir+'Karplus.txt', dtype=str)
print('Karplus relations', karplus_key)

# compute J coupling constants using our built-in funtion (compute_nonaa_Jcoupling) in toolbox.py
for i in range(100):    # 100 clustered states
    J = biceps.toolbox.compute_nonaa_Jcoupling(data_dir+'cineromycinB_pdbs/%d.fixed.pdb'%i, index=ind, karplus_key=karplus_key)
    np.savetxt(data_dir+'J_coupling/%d.txt'%i,J)


# Again, we need to convert computed J coupling constants to BICePs supported format.

# In[7]:


#########################################
# Lets' create input files for BICePs
############ Preparation ################
# Specify necessary argument values

# REQUIRED: raw data of pre-comuted chemical shifts
path = data_dir+'J_coupling/*txt'

# REQUIRED: number of states
states = 100

# REQUIRED: atom indices of pairwise distances
indices = data_dir+'atom_indice_J.txt'

# REQUIRED: experimental data
exp_data = data_dir+'exp_Jcoupling.txt'

# REQUIRED: topology file (as it only supports topology information, so it doesn't matter which state is used)
top = data_dir+'cineromycinB_pdbs/0.fixed.pdb'

# OPTIONAL: output directory of generated files
out_dir = data_dir+'noe_J'

p = biceps.Preparation('J',states=states,indices=indices,exp_data=exp_data,top=top,data_dir=path)   # 'J' scheme is selected
p.write(out_dir=out_dir)
# This will convert J coupling constants files for each state to a BICePs readable format and saved the new files in "noe_J" folder.


# Now, let's take a look at what's inside the newly generated files.

# In[8]:


fin = open(data_dir+'noe_J/0.J','r')
text = fin.read()
fin.close()
print(text)


# ### Conclusion###
# In this tutorial, we briefly showed how to use [Preparation](https://biceps.readthedocs.io/en/latest/api.html#preparation) class to prepare input files for BICePs using precomputed experimental observables. As of 2019, BICePs supports the following observables: NOE, J couplings, Chemical Shifts, Protection Factors. 
# 
# In the example above, we showed how to deal with NOE and J couplings (non-natural amino acids). 
# 
# For J couplings for natural amino acids, please check this tutorial. 
# 
# Chemical shifts can be computed using different algorithm. We recommend to use [Shiftx2](http://www.shiftx2.ca) which is also available in MDTraj library.  
# 
# Protection factors is a special observables which asks for extra work. We provide a [separate tutorial](https://biceps.readthedocs.io/en/latest/examples/Tutorials/Prep_Rest_Post_Ana/protectionfactors.html) that includes protection factors in BICePs sampling.
# 
# Now that the input files are ready, we can move on to [Restraint](https://biceps.readthedocs.io/en/latest/examples/Tutorials/Prep_Rest_Post_Ana/Restraint.html), where we construct a conformational ensemble. 
# 

# <h6 style="align: justify;font-size: 12pt"># <span style="color:red;">NOTE</span>: The following cell is for pretty notebook rendering</h6>

# In[9]:


from IPython.core.display import HTML
def css_styling():
    styles = open("../../../theme.css", "r").read()
    return HTML(styles)
css_styling()

