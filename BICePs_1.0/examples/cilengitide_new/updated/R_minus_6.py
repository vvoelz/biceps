#This program finds and saves rminus6 distances
import os, sys
import math
from scipy import loadtxt, savetxt
import numpy as np
from msmbuilder import Trajectory, io

usage = """Usage:   python Cluster_Variance.py input_name num1  
    Try:  python Cluster_Variance.py rgdf_nmev 100
"""
if len(sys.argv) < 2:
    print usage
    sys.exit(1)

input_name = sys.argv[1]
num1 = int(sys.argv[2])

input_numbers = [num1]

if __name__ == "__main__":

    traj = Trajectory.load_from_lhdf('Trajectories/trj0.lh5')

    #Load in all the pair data and add to dictionary#
def read_NOE_data(filename):

    os.system('''cat %s  | awk '{FS=" "}{print $3}' > atom1.txt'''%filename)
    os.system('''cat %s  | awk '{FS=" "}{print $4}' > atom2.txt'''%filename)
    os.system('''cat %s  | awk '{FS=" "}{print $8}' > pair_number.txt'''%filename)

    peptide_data = {}
    peptide_data['atom1'] = np.loadtxt('atom1.txt')
    peptide_data['atom2'] = np.loadtxt('atom2.txt')
    # Make sure that the pair_number tags start at 0
    peptide_data['pair_number'] = np.loadtxt('pair_number.txt') - 1
    
    return peptide_data

peptide_data = read_NOE_data('/Users/tud16919/Desktop/Analysis/NOE/%s'%input_name) 
atom1 = peptide_data['atom1']
atom2 = peptide_data['atom2']
distance_index_pairs = [(atom1,atom2)]
min_pairnum = int(peptide_data['pair_number'].min())
max_pairnum = int(peptide_data['pair_number'].max()) 
all_indices = np.arange(0,peptide_data['pair_number'].shape[0])

#Modified to only make files for cluster numbers not already in folder
for i in input_numbers:
    if not os.path.exists('Data/Assignments%d.h5' %i):
        os.system(' Cluster.py -a Data/Assignment%d.h5 -d Data/Assignments%d.h5.distances -g Data/Gens%d.lh5 rmsd hybrid -k %d' %(i,i,i,i))
    else:
        print 'Found Assignment%d.h5'%i, "Skipping."
        sys.exit(1)

#	*	*	*	*	Main Function	*	*	*	*	#
    	# Does a series of calculations for different numbers of clusters#


#Loop 1 ~ Uses Assignment files in the given range
for nclusters in input_numbers:   

    if (1):
        Assignments = io.loadh('Data/Assignment%d.h5'%nclusters)['arr_0']
        print nclusters, 'clusters. Shape of Assignments%d.h5 is'%nclusters, Assignments.shape
        ntraj, nframes = Assignments.shape
        natoms = traj["XYZList"].shape[1]  # (nframes, natoms, 3)

        #Loop 2 ~ Goes through the number of clusters one at a time for one number of clusters
        minus6avg = np.zeros((nclusters, len(range(min_pairnum, max_pairnum+1))))
        for k in range(0,nclusters):   # k is the index of the cluster
            Ind = (Assignments == k)[0,:] 

	    minus6distances = [[] for i in range(min_pairnum, max_pairnum+1)]# Compiles r**-6 distances for each pair number
            #Loop 3 ~ Finds all the distances for one pair number at a time
	    for pairnum in range(min_pairnum, max_pairnum+1):
	        # Finds the row indices of pairs with a given pair number
	        PairInd = (peptide_data['pair_number'] == pairnum)
	        indices = all_indices[PairInd]

		#Loop 4 ~ Computes distances and averages... 
  	        for i in indices:
                    atomnum_a, atomnum_b = peptide_data[ 'atom1' ][i], peptide_data[ 'atom2' ][i]
                    r1 = traj["XYZList"][Ind,atom1[i],:]  # arrays of *all* atom i positions
                    r2 = traj["XYZList"][Ind,atom2[i],:]

                    # Computes the distances scaled by r**-6
                    minus6distances[pairnum] += ((10.0*np.sqrt(np.sum((r1-r2)**2,axis=1)))**(-6.0)).tolist()   
                minus6avg[k, pairnum] = np.array(minus6distances[pairnum]).mean()
             
	    savetxt('minus6distances-for-%d.dat'%nclusters, minus6avg)
            print nclusters, 'clusters: minus6avg =', minus6avg
	    #print "minus6distances.shape:\t", np.array(minus6distances).shape, np.array(minus6distances)

    else:
        print 'Couldn\'t find Assignment%d.h5'%nclusters, "? Skipping."

        sys.exit(1)

