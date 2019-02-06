import os, sys, glob
import numpy as np

if len(sys.argv) < 2:
    print """Usage:  make_reduced_free_energies.py <epsilon>

    INPUT
    epsilon -- the strength of hydrophobic contacts in units kT (must be negative for attractive interaction) 

    Computes the reduced free energies f_i = n_i * epsilon/kT + ln(\Omega_i)
    of each contact state i, where:
        n_i is the number of contacts made in state i,
        \Omega_i is the number microstates in state i

    OUTPUT    
    reduced_free_energies_eps####.dat  where #### is the epsilon value
    """
    sys.exit(1)

eps = float(sys.argv[1])

# load in the data
contact_states, ncontacts, omegas = [], [], []

fin = open('contact_state_indices.dat','r')
lines = fin.readlines()
fin.close()

header = lines.pop(0) # pop the header line
for line in lines:
    fields = line.split('\t')
    contact_states.append(fields[1])  # repr of contact list
    ncontacts.append(float(fields[2]))
    omegas.append(float(fields[3]))

ncontacts = np.array(ncontacts)
omegas = np.array(omegas)

# compute the reduced free energies
f = ncontacts*eps + np.log(omegas)

outfile = 'reduced_free_energies_eps%4.3f.dat'%eps 
np.savetxt(outfile, f, fmt='%8.8f')
print 'Saved', outfile


