import itertools, glob
import os, sys, math
import mdtraj as md
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# this script will analyze all the pdbs that were generated using set_dihedrals.py
# and output the expected stereochemistry (based on the file-name), as well as what the
# stereochemistry is calculated to be.

all = []
traj = [md.load(i) for i in sorted(glob.glob("*pdb"))]
#traj = [md.load(i, top="xtc.gro") for i in sorted(glob.glob("data/*.xtc"))]
#indices = [[2,17,19,21],[21,23,70,72],[72,87,89,91],[91,93,140,142],[142,157,159,161],[161,163,0,2]] #h035
#indices = [[1,3,21,18],[18,17,54,55],[55,57,75,72],[72,71,108,109],[109,111,129,126],[126,125,0,1]] #g182
indices = [[1,3,21,18],[18,17,50,51],[51,53,71,68],[68,67,100,101],[101,103,121,118],[118,117,0,1]] #h030
choices = [[['c', 'c', 'c', 'c', 'c', 'c']],
 [['c', 'c', 'c', 'c', 'c', 't'],['c', 'c', 'c', 't', 'c', 'c'],['c', 't', 'c', 'c', 'c', 'c']],
 [['c', 'c', 'c', 'c', 't', 'c'],['c', 'c', 't', 'c', 'c', 'c'],['t', 'c', 'c', 'c', 'c', 'c']],
 [['c', 'c', 'c', 'c', 't', 't'],['c', 'c', 't', 't', 'c', 'c'],['t', 't', 'c', 'c', 'c', 'c']],
 [['c', 'c', 'c', 't', 't', 'c'],['c', 't', 't', 'c', 'c', 'c'],['t', 'c', 'c', 'c', 'c', 't']],
 [['c', 'c', 'c', 't', 'c', 't'],['c', 't', 'c', 't', 'c', 'c'],['c', 't', 'c', 'c', 'c', 't']],
 [['c', 'c', 't', 'c', 't', 'c'],['t', 'c', 'c', 'c', 't', 'c'],['t', 'c', 't', 'c', 'c', 'c']],
 [['c', 'c', 't', 'c', 'c', 't'],['t', 'c', 'c', 't', 'c', 'c']],
 [['c', 't', 'c', 'c', 't', 'c']],
 [['c', 'c', 'c', 't', 't', 't'],['c', 't', 't', 't', 'c', 'c'],['t', 't', 'c', 'c', 'c', 't']],
 [['c', 'c', 't', 't', 't', 'c'],['t', 'c', 'c', 'c', 't', 't'],['t', 't', 't', 'c', 'c', 'c']],
 [['c', 'c', 't', 'c', 't', 't'],['t', 'c', 't', 't', 'c', 'c'],['t', 't', 'c', 'c', 't', 'c']],
 [['c', 't', 'c', 't', 't', 'c'],['c', 't', 't', 'c', 'c', 't'],['t', 'c', 'c', 't', 'c', 't']], 
 [['c', 'c', 't', 't', 'c', 't'],['c', 't', 'c', 'c', 't', 't'],['t', 't', 'c', 't', 'c', 'c']],
 [['c', 't', 't', 'c', 't', 'c'],['t', 'c', 'c', 't', 't', 'c'],['t', 'c', 't', 'c', 'c', 't']],
 [['c', 't', 'c', 't', 'c', 't']],
 [['t', 'c', 't', 'c', 't', 'c']],
 [['c', 'c', 't', 't', 't', 't'],['t', 't', 'c', 'c', 't', 't'],['t', 't', 't', 't', 'c', 'c']],
 [['c', 't', 't', 't', 't', 'c'],['t', 'c', 'c', 't', 't', 't'],['t', 't', 't', 'c', 'c', 't']],
 [['c', 't', 'c', 't', 't', 't'],['c', 't', 't', 't', 'c', 't'],['t', 't', 'c', 't', 'c', 't']],
 [['t', 'c', 't', 'c', 't', 't'],['t', 'c', 't', 't', 't', 'c'],['t', 't', 't', 'c', 't', 'c']],
 [['c', 't', 't', 'c', 't', 't']],
 [['t', 'c', 't', 't', 'c', 't'],['t', 't', 'c', 't', 't', 'c']],
 [['c', 't', 't', 't', 't', 't'],['t', 't', 'c', 't', 't', 't'],['t', 't', 't', 't', 'c', 't']],
 [['t', 'c', 't', 't', 't', 't'],['t', 't', 't', 'c', 't', 't'],['t', 't', 't', 't', 't', 'c']],
 [['t', 't', 't', 't', 't', 't']]]


with open('reweight.txt') as file:
    lines = file.readlines()

for i in range(len(lines)):
    lines[i] = lines[i].strip()
    lines[i] = lines[i].split()
    lines[i][0] = int(lines[i][0])
    lines[i][1] = float(lines[i][1])

x_ticks = [''.join(choices[i][0]) for i in range(len(choices))]

dihedrals = []
for i in range(len(traj)):
    dihedrals.append(md.compute_dihedrals(traj[i], np.asarray(indices)))

best_dihedrals = []
results = [0]*len(choices)
for i in range(len(dihedrals)): # each trajectory
    for j in range(len(dihedrals[i])): # each frame
        best_dihedrals = []
        for k in range(len(dihedrals[i][j])): # each angle
            if dihedrals[i][j][k] >= -1.57 and dihedrals[i][j][k] < 1.57:
                 best_dihedrals.append("c")
            elif dihedrals[i][j][k] < -1.57 or dihedrals[i][j][k] >= 1.57:
                 best_dihedrals.append("t")
                    
        for k in range(len(choices)):
            if best_dihedrals in choices[k]:
                results[k] += 1000**lines[i][1]
#                results[k] += -math.log(0.999893+lines[i][1])

results = results / np.linalg.norm(results)

#for i in range(len(results)):
#    if results[i] > 0:
#        print "".join(choices[i][0]), results[i]
#    else:
#        print "".join(choices[i][0]), "0"

plt.figure(figsize=(10,5))
plt.bar(range(len(x_ticks)), results, align='center')
plt.xticks(range(len(x_ticks)), x_ticks, rotation=90)
plt.title("H030 REMD Omega Angle Distribution")
plt.tight_layout()
plt.savefig('neglog_H030_omega.png')

with open("x_3","w+") as x_2:
    for i in range(len(x_ticks)):
        x_2.write(str(i) + "\n")
    x_2.close()
with open("y_3","w+") as y_2:
    for i in results:
        y_2.write(str(i) + "\n")
    y_2.close()
with open("z_3","w+") as z_2:
    for i in x_ticks:
        z_2.write(str(i) + "\n")
    z_2.close()

