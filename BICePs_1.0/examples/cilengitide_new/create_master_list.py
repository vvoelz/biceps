import os, sys, string
import numpy as np

from scipy import loadtxt, savetxt

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

fontfamily={'family':'sans-serif','sans-serif':['Arial']}
plt.rc('font', **fontfamily)




class DataElement(object):
    """A class to store all the relevant data for an atom pair.

#atom1  #atom2  #1      #2      #avg    #low    #up     #pair   #1      #2
ARG1HN	ARG1HA	1	3	0	217	300	1	
ARG1HN	ARG1HB	1	5 	0	265	322	2		
ARG1HN  ARG1HB  1       6       0       265     322     2
ARG1HN	ARG1HG	1	8	0	271	346	3		
"""

    def __init__(self, i, j, name1=None, name2=None, avg_distance=None, lower_bound=None, upper_bound=None, pair_index=None, comment=''):
        """All distances are given in angstroms."""

        self.i = i
        self.j = j 
        self.name1 = name1
	self.name2 = name2
        self.avg_distance = avg_distance
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.pair_index = pair_index
        self.comment = comment

    def header(self):
        """returns a header string."""
        headings = ['atomInd1', 'atomInd2', 'name1', 'name2', 'avg_distance', 'lower_bound', 'upper_bound', 'pair_index', 'comment']
        return '# '+ string.joinfields(headings,'\t')

    def __repr__(self):
        """Returns a string representation of the line."""

        return "%d\t%d\t%-8s\t%-8s\t%3.3f\t%3.3f\t%3.3f\t%d\t%s"%(self.i, self.j, self.name1, self.name2, self.avg_distance, self.lower_bound, self.upper_bound,
                  self.pair_index, self.comment)


######################################
#  Main
######################################

data = {}   # a dict of DataElement() objects keyed by (i,j) atom indices

# Read in table values from previous file
fin = open('rgdf_nmev_NOE_Data_VAV.dat','r')
lines = fin.readlines()
fin.close()

for line in lines:
    if line[0] != '#':
        fields = line.split()
        s1, s2 = fields[0], fields[1]
        i, j = int(fields[2]), int(fields[3])
        avg = float(fields[4])/100.
        low, up = float(fields[5])/100., float(fields[6])/100.
        pair_index = int(fields[7])
        remark = string.joinfields(fields[8:])

        # put entry in data table
        data[ (i,j) ] = DataElement(i, j, name1=s1, name2=s2, avg_distance=avg, lower_bound=low, upper_bound=up, pair_index=pair_index, comment=remark)

sorted_keys = data.keys()
sorted_keys.sort()
print data[sorted_keys[0]].header()
for k in sorted_keys:
    print data[k]


atom1 = loadtxt('updated_VAV/atom1.txt').astype(int)
atom2 = loadtxt('updated_VAV/atom2.txt').astype(int)
print 'atom1.shape', atom1.shape
print 'atom2.shape', atom2.shape

minus6_keys = [(atom1[k],atom2[k]) for k in range(len(atom1))]
print 'minus6_keys', minus6_keys, 'len(minus6_keys)', len(minus6_keys)

minus6_avg_simdist = loadtxt('updated_VAV/minus6distances-for-100.dat')**(-1./6.)
print 'minus6_avg_simdist.shape', minus6_avg_simdist.shape

print data[minus6_keys[0]].header()
for k in range(len(minus6_keys)):
    atompair = minus6_keys[k]
    pair_index = data[atompair].pair_index       
    print minus6_keys[k], minus6_avg_simdist[:,pair_index-1]  #note pair indices  start at 1 (!)

print '##########'

# Find agreement with all states
R2_values_bystate = []
for state in range(100):
  exp_values, sim_values = [],[]
  for k in range(len(minus6_keys)):
    atompair = minus6_keys[k]
    pair_index = data[atompair].pair_index

    if (0):
        # r^-6-average dist 
        data[atompair].avg_distance = ( ((data[atompair].lower_bound)**(-6.) + (data[atompair].upper_bound)**(-6.))/2.0  )**(-1./6.)   
    else:
        # mean dist
        data[atompair].avg_distance = ( data[atompair].lower_bound + data[atompair].upper_bound)/2.0

    print minus6_keys[k], 'sim', minus6_avg_simdist[state,pair_index-1], 'exp:', data[atompair].avg_distance
    exp_values.append( data[atompair].avg_distance )
    sim_values.append( minus6_avg_simdist[state,pair_index-1] )
  R2 = np.corrcoef(exp_values, sim_values)[0,1]
  R2_values_bystate.append( R2 )

print '########## print revised data table ###########'
print data[sorted_keys[0]].header()
for k in sorted_keys:
    print data[k]

print '########## print YAML-style output ###########'
NOE_Expt = [data[k].avg_distance for k in sorted_keys]
print 'NOE_Expt:', NOE_Expt
print 'NOE_PairIndex:'
for k in sorted_keys:
    print '- [%d, %d]'%k


from matplotlib import pyplot as plt

if (0):
    plt.figure(figsize=(4,4))
    for state in range(100):
        plt.plot(state, R2_values_bystate[state], '*')
        plt.text(state, R2_values_bystate[state], str(state))
    plt.show()

if (1):
    plt.figure(figsize=(4,3))
    ax = plt.axes(frameon=False)
    ax.set_xticks([]) 
    #ax.set_yticks([]) 
    R2_values_bystate = np.array(R2_values_bystate)
    Ind = np.argsort(-R2_values_bystate) 
    top_indices = [53,78,60,72,75]
    plt.plot(range(Ind.shape[0]), R2_values_bystate[Ind], 'k-', linewidth=2)
    for rank in range(Ind.shape[0]):
      if top_indices.count(Ind[rank]) > 0:
        plt.plot(rank,R2_values_bystate[Ind[rank]],'g*', markersize=14)
        plt.text(rank*1.5,R2_values_bystate[Ind[rank]]+0.05, str(Ind[rank]), color='g', fontsize=8)
        plt.plot([rank,rank],[np.min(R2_values_bystate),R2_values_bystate[Ind[rank]]],'g-')
    plt.xlabel('rank')
    plt.ylabel('$R^2$')
    plt.tight_layout()
    plt.savefig('ranking.pdf')

if (0):
    plt.figure(figsize=(4,4))
    plt.plot(exp_values, sim_values, '*')
    plt.plot([2,5],[2,5],'k-')
    plt.show()




