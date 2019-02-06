import sys, os, glob

sys.path.append('../../src')

from Structure import *
from PosteriorSampler import *

import numpy as np
from scipy import loadtxt, savetxt
import yaml
import cPickle, pprint

from pymbar import MBAR

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("popsfile1", help="the path name of the population *.dat file")
parser.add_argument("popsfile2", help="the path name of the population *.dat file")

args = parser.parse_args()

print '=== Settings ==='
print 'popsfile1', args.popsfile1
print 'popsfile2', args.popsfile2


K = 3

# Load in precomputed P and dP from calc_MBAR_fromresults....py
# >>> python plot_pops_vs_PCM.py populations_ref_normal.dat ../cineromycin_B/populations_ref_normal.dat
P_dP_PCM = loadtxt(args.popsfile1)
P_dP = loadtxt(args.popsfile2)

pops0, pops1   = P_dP[:,0], P_dP[:,K-1]
dpops0, dpops1 = P_dP[:,K], P_dP[:,2*K-1]

pops0_PCM, pops1_PCM   = P_dP_PCM[:,0], P_dP_PCM[:,K-1]
dpops0_PCM, dpops1_PCM = P_dP_PCM[:,K], P_dP_PCM[:,2*K-1]

print 'group\tpop\tpop + PCM'
groupA = np.array([pops1[i] for i in [38, 39, 65, 90]]).sum()
groupA_PCM = np.array([pops1_PCM[i] for i in [38, 39, 65, 90]]).sum()
print 'A\t%f\t%f'%( groupA, groupA_PCM)

groupB = np.array([pops1[i] for i in [45, 59]]).sum()
groupB_PCM = np.array([pops1_PCM[i] for i in [45, 59]]).sum()
print 'B\t%f\t%f'%( groupB, groupB_PCM)

groupC = np.array([pops1[i] for i in [80, 85]]).sum()
groupC_PCM = np.array([pops1_PCM[i] for i in [80, 85]]).sum()
print 'C\t%f\t%f'%( groupC, groupC_PCM)

groupD = pops1[92]
groupD_PCM = pops1_PCM[92]
print 'D\t%f\t%f'%( groupD, groupD_PCM)

sys.exit(1)



######## PLOTTING ########


# Figure Plot SETTINGS
label_fontsize = 16
legend_fontsize = 14

# Make a figure
plt.figure()

# Make a subplot in the upper left
if (1):
    plt.subplot(2,2,1)
    # We assume: column 0 is lambda=0.0 and column K-1 is lambda=1.0
    plt.errorbar( pops1_PCM, pops1, xerr=dpops1_PCM, yerr=dpops1, fmt='k.')
    plt.hold(True)
    plt.plot([1e-5, 1], [1e-5, 1], color='k', linestyle='-', linewidth=2)
    plt.xlim(1e-5, 1.)
    plt.ylim(1e-5, 1.)
    plt.xlabel('$p_i$ (+PCM)', fontsize=label_fontsize)
    plt.ylabel('$p_i$ ', fontsize=label_fontsize)
    plt.xscale('log')
    plt.yscale('log')
    # label key states
    plt.hold(True)
    #for i in [87, 21, 79]:
    #    plt.text( pops1_PCM[i], pops1[i], str(i), color='r' )
    #for i in [38, 39, 45, 59, 65, 80, 85, 90, 92]:
    #    plt.text( pops1_PCM[i], pops1[i], str(i), color='g' )

    #plt.tight_layout()

# Read in populations from the original DFT estimates
energies_PCM = loadtxt('cineromycinB_QMenergies_PCM.dat')*627.509  # convert from hartrees to kcal/mol
energies_PCM = energies_PCM/0.5959   # convert to reduced free energies F = f/kT
energies_PCM -= energies_PCM.min()
QM_pops_PCM = np.exp(-energies_PCM)
QM_pops_PCM = QM_pops_PCM/QM_pops_PCM.sum()

energies = loadtxt('../cineromycin_B/cineromycinB_QMenergies.dat')*627.509  # convert from hartrees to kcal/mol
energies = energies/0.5959   # convert to reduced free energies F = f/kT
energies -= energies.min()
QM_pops = np.exp(-energies)
QM_pops = QM_pops/QM_pops.sum()


# QM_pops versus pops (QM+exp)
if (1):
    # Make a subplot in the upper right 
    plt.subplot(2,2,3)
    # We assume: column 0 is lambda=0.0 and column K-1 is lambda=1.0
    plt.errorbar( QM_pops, pops1, yerr=dpops1, fmt='k.')
    plt.hold(True)
    plt.plot([1e-5, 1], [1e-5, 1], color='k', linestyle='-', linewidth=2)
    plt.xlim(1e-5, 1.)
    plt.ylim(1e-5, 1.)
    plt.xlabel('$p_i$ (QM only) ', fontsize=label_fontsize)
    plt.ylabel('$p_i$ (QM+exp)', fontsize=label_fontsize)
    plt.xscale('log')
    plt.yscale('log')
    # label key states
    plt.hold(True)
    #for i in [87, 21, 79]:
    #    plt.text( pops1_PCM[i], pops1[i], str(i), color='r' )
    #for i in [38, 39, 45, 59, 65, 80, 85, 90, 92]:
    #    plt.text( pops1_PCM[i], pops1[i], str(i), color='g' )

# QM_pops+PCM versus pops (QM+exp) +PCM
if (1):
    # Make a subplot in the upper right 
    plt.subplot(2,2,4)
    # We assume: column 0 is lambda=0.0 and column K-1 is lambda=1.0
    plt.errorbar( QM_pops_PCM, pops1_PCM, yerr=dpops1_PCM, fmt='k.')
    plt.hold(True)
    plt.plot([1e-5, 1], [1e-5, 1], color='k', linestyle='-', linewidth=2)
    plt.xlim(1e-5, 1.)
    plt.ylim(1e-5, 1.)
    plt.xlabel('$p_i$+PCM (QM only)', fontsize=label_fontsize)
    plt.ylabel('$p_i$+PCM (QM+exp)', fontsize=label_fontsize)
    plt.xscale('log')
    plt.yscale('log')
    # label key states
    plt.hold(True)
    #for i in [87, 21, 79]:
    #    plt.text( pops1_PCM[i], pops1[i], str(i), color='r' )
    #for i in [38, 39, 45, 59, 65, 80, 85, 90, 92]:
    #    plt.text( pops1_PCM[i], pops1[i], str(i), color='g' )

    #plt.tight_layout()





plt.show()




