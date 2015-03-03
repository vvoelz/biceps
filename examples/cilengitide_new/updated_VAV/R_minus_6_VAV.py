#This program finds and saves rminus6 distances
import os, sys
import math
from scipy import loadtxt, savetxt
import numpy as np
#from msmbuilder import Trajectory, io
import mdtraj
from mdtraj import io

usage = """Usage:   python R_minus_6_VAV.py

"""


#### Functions ########

def read_NOE_data():

    txt = """# atomInd1	atomInd2	name1	name2	avg_distance	lower_bound	upper_bound	pair_index	comment
1	3	ARG1HN  	ARG1HA  	2.382	2.170	3.000	1	
1	5	ARG1HN  	ARG1HB  	2.843	2.650	3.220	2	
1	6	ARG1HN  	ARG1HB  	2.843	2.650	3.220	2	
1	8	ARG1HN  	ARG1HG  	2.938	2.710	3.460	3	
1	9	ARG1HN  	ARG1HG  	2.938	2.710	3.460	3	
1	25	ARG1HN  	GLY2HN  	3.364	3.130	3.830	7	
1	71	ARG1HN  	VAL5HA  	2.702	2.460	3.420	8	
1	72	ARG1HN  	VAL5HB  	2.898	2.660	3.490	9	
1	73	ARG1HN  	VAL5C   	3.108	2.860	3.700	10	#PROrHG
1	74	ARG1HN  	VAL5C   	0.000	2.860	3.700	10	#PROrHG
1	75	ARG1HN  	VAL5C   	0.000	2.860	3.700	10	#PROrHG
1	76	ARG1HG  	VAL5C   	2.436	2.220	3.060	12	#proRHG
1	77	ARG1HG  	VAL5C   	2.436	2.220	3.060	12	#proRHG
1	78	ARG1HG  	VAL5C   	2.436	2.220	3.060	12	#proRHG
1	79	ARG1HN  	VAL5N   	2.815	2.620	3.200	11	#(CH3)
1	80	ARG1HN  	VAL5N   	2.815	2.620	3.200	11	#(CH3)
1	81	ARG1HN  	VAL5N   	2.815	2.620	3.200	11	#(CH3)
3	5	ARG1HA  	ARG1HB  	2.331	2.170	2.650	4	
3	6	ARG1HA  	ARG1HB  	2.331	2.170	2.650	4	
3	8	ARG1HA  	ARG1HG  	2.422	2.240	2.820	5	
3	9	ARG1HA  	ARG1HG  	2.422	2.240	2.820	5	
5	11	ARG1HB  	ARG1HD  	2.138	1.990	2.430	6	
5	12	ARG1HB  	ARG1HD  	2.138	1.990	2.430	6	
6	11	ARG1HB  	ARG1HD  	2.138	1.990	2.430	6	
6	12	ARG1HB  	ARG1HD  	2.138	1.990	2.430	6	
25	3	GLY2HN  	ARG1HA  	2.653	2.430	3.230	13	
25	5	GLY2HN  	ARG1HB  	3.279	2.990	4.100	14	
25	6	GLY2HN  	ARG1HB  	3.279	2.990	4.100	14	
25	8	GLY2HN  	ARG1HG  	4.395	4.090	5.000	15	
25	9	GLY2HN  	ARG1HG  	4.395	4.090	5.000	15	
25	27	GLY2HN  	GLY2HA  	2.651	2.420	3.290	16	#PROr
25	28	GLY2HN  	GLY2HA  	2.996	2.750	3.610	17	#PROs
25	79	GLY2HN  	VAL5N   	4.192	3.850	5.030	18	#(CH3)
25	80	GLY2HN  	VAL5N   	4.192	3.850	5.030	18	#(CH3)
25	81	GLY2HN  	VAL5N   	4.192	3.850	5.030	18	#(CH3)
32	27	ASP3HN  	GLY2HA  	2.866	2.600	3.730	19	#PROr
32	28	ASP3HN  	GLY2HA  	2.640	2.410	3.280	20	#PROs
32	34	ASP3HN  	ASP3HA  	2.751	2.560	3.130	21	
32	36	ASP3HN  	ASP3HB  	3.033	2.770	3.760	22	#PROr
32	37	ASP3HN  	ASP3HB  	3.250	2.970	4.010	23	#PROs
32	44	ASP3HN  	PHE4HN  	3.859	3.550	4.600	24	
32	79	ASP3HN  	VAL5N   	4.146	3.840	4.800	25	#(CH3)
32	80	ASP3HN  	VAL5N   	4.146	3.840	4.800	25	#(CH3)
32	81	ASP3HN  	VAL5N   	4.146	3.840	4.800	25	#(CH3)
44	27	PHE4HN  	GLY2HA  	4.212	3.920	4.790	26	#PROr
44	28	PHE4HN  	GLY2HA  	0.000	3.920	4.790	26	#PROr
44	34	PHE4HN  	ASP3HA  	2.310	2.150	2.630	27	
44	36	PHE4HN  	ASP3HB  	4.092	3.810	4.650	28	#PROr
44	37	PHE4HN  	ASP3HB  	3.985	3.710	4.530	29	#PROs
44	46	PHE4HN  	PHE4HA  	2.978	2.750	3.490	30	
44	48	PHE4HN  	PHE4HB  	2.469	2.270	2.950	31	
44	49	PHE4HN  	PHE4HB  	2.469	2.270	2.950	31	
44	52	PHE4HN  	PHE4HD  	3.621	3.340	4.260	32	
44	71	PHE4HN  	VAL5HA  	4.512	4.200	5.130	34	
44	79	PHE4HN  	VAL5N   	3.570	3.270	4.350	35	#(CH3)
44	80	PHE4HN  	VAL5N   	3.570	3.270	4.350	35	#(CH3)
44	81	PHE4HN  	VAL5N   	3.570	3.270	4.350	35	#(CH3)
46	48	PHE4HA  	PHE4HB  	2.364	2.150	3.010	33	
46	49	PHE4HA  	PHE4HB  	2.364	2.150	3.010	33	
46	79	PHE4HA  	VAL5N   	2.289	2.100	2.760	36	#(CH3)
46	80	PHE4HA  	VAL5N   	2.289	2.100	2.760	36	#(CH3)
46	81	PHE4HA  	VAL5N   	2.289	2.100	2.760	36	#(CH3)
48	52	PHE4HB  	PHE4HD  	2.265	2.110	2.570	37	
49	52	PHE4HB  	PHE4HD  	2.265	2.110	2.570	37	
71	72	VAL5HA  	VAL5HB  	2.852	2.630	3.360	38	
71	73	VAL5HA  	VAL5C   	2.402	2.210	2.860	40	#PROSHG
71	74	VAL5HA  	VAL5C   	2.402	2.210	2.860	40	#PROSHG
71	75	VAL5HA  	VAL5C   	2.402	2.210	2.860	40	#PROSHG
71	76	VAL5HA  	VAL5C   	2.413	2.200	3.020	39	#PRORHG
71	77	VAL5HA  	VAL5C   	2.413	2.200	3.020	39	#PRORHG
71	78	VAL5HA  	VAL5C   	2.413	2.200	3.020	39	#PRORHG
72	79	VAL5HB  	VAL5N   	2.082	1.920	2.450	41	#(CH3)
72	80	VAL5HB  	VAL5N   	2.082	1.920	2.450	41	#(CH3)
72	81	VAL5HB  	VAL5N   	2.082	1.920	2.450	41	#(CH3)"""


    lines = txt.split('\n')
    lines.pop(0)  # remove the header
    print lines

    peptide_data = {}
    peptide_data['atom1'] = np.array( [int(line.split('\t')[0]) for line in lines] )
    peptide_data['atom2'] = np.array( [int(line.split('\t')[1]) for line in lines] )

    # Make sure that the pair_number tags start at 0
    peptide_data['pair_number'] = np.array( [int(line.split('\t')[7])-1 for line in lines] )
    
    return peptide_data



########## Main #########


#if __name__ == "__main__":
if (1):
    #traj = mdtraj.load('/Users/vv/data/cilengitide_cistrans_remd_2.4us/analysis/Trajectories/trj0.lh5')
    traj = mdtraj.load_xtc('/Users/vv/data/cilengitide_cistrans_remd_2.4us/remd/traj0.xtc', 
                           top='/Users/vv/data/cilengitide_cistrans_remd_2.4us/remd/conf_prod0.pdb')

    #Load in all the pair data and add to dictionary#
    peptide_data = read_NOE_data()
    print "peptide_data['atom1']", peptide_data['atom1']
    print "peptide_data['atom1']", peptide_data['atom2']
    print "peptide_data['pair_number']", peptide_data['pair_number']

    savetxt('atom1.txt', peptide_data['atom1'])
    savetxt('atom2.txt', peptide_data['atom2'])
    savetxt('pair_number.txt', peptide_data['pair_number'])

    min_pairnum = peptide_data['pair_number'].min()
    max_pairnum = peptide_data['pair_number'].max() 
    all_indices = np.arange(0,peptide_data['pair_number'].shape[0])

    # Load assignments file
    assignFn = '../updated/Assignment100.h5'
    Assignments = io.loadh(assignFn)['arr_0']
    nclusters = 100
    print 'shape of %s is'%assignFn, Assignments.shape
    ntraj, nframes = Assignments.shape
    natoms = traj.xyz.shape[1]  # (nframes, natoms, 3)

    #Loop 2 ~ Goes through the number of clusters one at a time for one number of clusters
    minus6avg = np.zeros((nclusters, len(range(min_pairnum, max_pairnum+1))))
    for k in range(0,nclusters):   # k is the index of the cluster

        # get indices of snapshot corresponding to cluster k
        Ind = (Assignments == k)[0,:] 

        minus6distances = [[] for i in range(min_pairnum, max_pairnum+1)]# Lists to store  r**-6 distances for each pair number

        #Loop 3 ~ Finds all the distances for one pair number at a time
        for pairnum in range(min_pairnum, max_pairnum+1):
	        # Finds the row indices of pairs with a given pair number
	        PairInd = (peptide_data['pair_number'] == pairnum)
	        indices = all_indices[PairInd]

		#Loop 4 ~ Computes distances and averages... 
  	        for i in indices:
                    atomnum_a, atomnum_b = peptide_data[ 'atom1' ][i], peptide_data[ 'atom2' ][i]
                    r1 = traj.xyz[Ind,peptide_data['atom1'][i],:]  # arrays of *all* atom i positions
                    r2 = traj.xyz[Ind,peptide_data['atom2'][i],:]

                    # Computes the distances scaled by r**-6
                    minus6distances[pairnum] += ((10.0*np.sqrt(np.sum((r1-r2)**2,axis=1)))**(-6.0)).tolist()   
                minus6avg[k, pairnum] = np.array(minus6distances[pairnum]).mean()
             
    savetxt('minus6distances-for-%d.dat'%nclusters, minus6avg)
    print nclusters, 'clusters: minus6avg =', minus6avg
    #print "minus6distances.shape:\t", np.array(minus6distances).shape, np.array(minus6distances)

