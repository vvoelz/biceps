import sys, os, glob

sys.path.append('../../src')

from Structure import *
from PosteriorSampler import *

import cPickle  # to read/write serialized sampler classes

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("lam", help="a lambda value between 0.0 and 1.0  denoting the Hamiltonian weight (E_data + lambda*E_QM)", type=float)
parser.add_argument("outdir", help="the name of the output directory")
parser.add_argument("nsteps", help="Number of sampling steps", type=int)
parser.add_argument("--noref", help="Do not use reference potentials (default is to use them)",
                    action="store_true")
parser.add_argument("--lognormal", help="Use log-normal distance restraints (default is normal)",
                    action="store_true")
parser.add_argument("--verbose", help="use verbose output",
                    action="store_true")
args = parser.parse_args()


print '=== Settings ==='
print 'lam', args.lam
print 'outdir', args.outdir
print 'nsteps', args.nsteps
print '--noref', args.noref
print '--lognormal', args.lognormal
print '--verbose', args.verbose



"""
OUTPUT 

    Files written:
        <outdir>/traj_lambda_<lambda>.yaml  - YAML Trajectory file 
        <outdit>/sampler_<lambda>.pkl       - a cPickle'd sampler object


RUNME:
>>> python plot_predicted_observables.py 1.0 results_ref_normal 1000
"""

# Make a new directory if we have to
if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)


# Functions

def distance_info():
    """Returns a lists of all NOE_names and a list of membership (index of NOE_names) for each distance restraint."""

    NOE_names = ['H(3)/H(6)', 'H(3)/H(9)', 'H(6)/H(9)', 'H(5)/H(7)', 'H(6)/H(7)', 'H(7)/H(9)', 'H(9)/H(12)', 
                 "H(9)/H(11')", "Me(12)/H(10')", "Me(6)/H(6)", "Me(8)/H(7)", "Me(12)/H(13)", "Me(12)/Me(13)"]
    NOE_membership = []

    raw = """- [23, 25]    0    H(3)/H(6)
- [23, 27]    1    H(3)/H(9)
- [25, 27]    2    H(6)/H(9) 
- [24, 26]    3    H(5)/H(7)
- [25, 26]    4    H(6)/H(7)
- [26, 27]    5    H(7)/H(9) 
- [27, 32]    6    H(9)/H(12)
- [27, 30]    7    H(9)/H(11')
- [27, 31]    7    " 
- [28, 38]    8    Me(12)/H(10') 
- [28, 39]    8    "
- [28, 40]    8    "
- [29, 38]    8    "
- [29, 39]    8    " 
- [29, 40]    8    "
- [25, 41]    9    Me(4)/H(6)
- [25, 42]    9    "
- [25, 43]    9    "
- [26, 44]    10   Me(8)/H(7)
- [26, 45]    10   "
- [26, 46]    10   "
- [33, 38]    11   Me(12)/H(13)
- [33, 39]    11   "
- [33, 40]    11   "
- [35, 38]    12   Me(12)/Me(13)
- [35, 39]    12
- [35, 40]    12
- [36, 38]    12
- [36, 39]    12
- [36, 40]    12
- [37, 38]    12
- [37, 39]    12
- [37, 40]    12   """

    rawlines = raw.split('\n')
    for line in rawlines:
        print 'line[14:17]', line[14:17]
        NOE_membership.append(int(line[14:17]))

    return NOE_names, NOE_membership


def dihedral_info():
    """Returns a list of J_names and list of membership."""

    J_names = ["$^3J_{6,7}$", "$^3J_{11,12}$", "$^3J_{Me,H(12)}$", "$^3J_{12,13}$", "$^3J_{Me,H(13)}$"]
    J_membership = []

    raw = """- [25,9,10,26]    0    J_6,7
- [30,14,15,32]    1    J_11,12
- [31,14,15,32]    1    "
- [32,15,20,38]    2    J_Me,H(12)
- [32,15,20,39]    2    "
- [32,15,20,40]    2    "
- [32,15,16,33]    3    J_12,13
- [33,16,21,35]    4    J_Me,H(13)
- [33,16,21,36]    4    "
- [33,16,21,37]    4      """

    rawlines = raw.split('\n')
    print rawlines
    for line in rawlines:
        print 'line[18:21]', line[18:21]
        J_membership.append(int(line[18:21]))

    return J_names, J_membership




#########################################
# Let's create our ensemble of structures

expdata_filename = 'cineromycinB_expdata_VAV.yaml'
#expdata_filename = 'cineromycinB_expdata.yaml'  <-- doesn't contain the right Karplus specs 
energies_filename = 'cineromycinB_QMenergies_PCM.dat'
energies = loadtxt(energies_filename)*627.509  # convert from hartrees to kcal/mol
energies = energies/0.5959   # convert to reduced free energies F = f/kT
energies -= energies.min()

ensemble = []
for i in range(100):

    print
    print '#### STRUCTURE %d ####'%i

    # no information from QM --> lam = 0.0
    # QM + exp               --> lam = 1.0
    ensemble.append( Structure('cineromycinB_pdbs/%d.fixed.pdb'%i, args.lam*energies[i], expdata_filename, use_log_normal_distances=False) )

    


if (1):
  # Load in precomputed P and dP from calc_MBAR_fromresults....py
  # >>> python plot_pops_vs_PCM.py populations_ref_normal.dat ../cineromycin_B/populations_ref_normal.dat
  P_dP_PCM = loadtxt('populations_ref_normal.dat')
  K = 3

  equilpops = P_dP_PCM[:,K-1]
  groupA = [38, 39, 65, 90]
  groupB = [45, 59]
  groupC = [80, 85]
  groupD = [92]

  NOE_names, NOE_membership = distance_info()
  print "NOE_names, NOE_membership"
  print NOE_names, NOE_membership

  J_names, J_membership = dihedral_info()
  print "J_names, J_membership"
  print J_names, J_membership

  # collect distance distributions for each distance and plot them for selected states
  s = ensemble[0]
  ndistances = len(s.distance_restraints)
  ndihedrals = len(s.dihedral_restraints)
  all_distances = []
  distance_distributions = [[] for i in range(ndistances)]
  for s in ensemble:
    for i in range(ndistances):
        distance_distributions[i].append( s.distance_restraints[i].model_distance )
        all_distances.append( s.distance_restraints[i].model_distance )

  colors = {0:'ro', 1:'go', 2:'bo', 3:'co'}
  linecolors = {0:'r-', 1:'g-', 2:'b-', 3:'c-'}
  groups = [groupA, groupB, groupC, groupD]
  for g in range(len(groups)):

        plt.figure(figsize=(8,4))
        group = groups[g] 

        # plot distances
        if (1):
            ax = plt.subplot(1, 2, 1)
            d_mle, gamma_mle = 2.5, 1.25
            plt.plot( [0,len(NOE_names)],[d_mle*gamma_mle, d_mle*gamma_mle], 'k-', linewidth=2)
            plt.hold(True)
            for k in group:
              for i in range(ndistances):
                weight = ensemble[k].distance_restraints[i].weight
                print i, ensemble[k].distance_restraints[i].model_distance, weight
                x = NOE_membership[i]
                d = ensemble[k].distance_restraints[i].model_distance
                plt.plot([x-0.4,x+0.4], [d,d], linecolors[g], linewidth=3.0*weight)
                plt.hold(True)
            ax.set_xticks(range(len(NOE_names)))
            ax.set_xticklabels(NOE_names, rotation=90)

            plt.ylabel('distance ($\AA$)')

        # plot J-coupling
        if (1):
            ax = plt.subplot(1, 2, 2)
            for k in group:
              for i in range(ndihedrals):
                weight = ensemble[k].dihedral_restraints[i].weight
                'print i, ensemble[k].dihedral_restraints[i].model_Jcoupling, ensemble[k].dihedral_restraints[i].exp_Jcoupling, weight'
                print i, ensemble[k].dihedral_restraints[i].model_Jcoupling, ensemble[k].dihedral_restraints[i].exp_Jcoupling, weight
                model, exp = ensemble[k].dihedral_restraints[i].model_Jcoupling, ensemble[k].dihedral_restraints[i].exp_Jcoupling
                x = J_membership[i]
                plt.plot([x-0.4,x+0.4], [exp, exp], 'k-')
                plt.plot([x-0.4,x+0.4], [model, model], linecolors[g])
                plt.hold(True)
            ax.set_xticks(range(len(J_names)))
            ax.set_xticklabels(J_names, rotation=90)

            plt.ylabel('$^3J_{HH}$')

        plt.tight_layout()
        plt.show()

# for debugging
sys.exit(1)

