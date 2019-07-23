#!/usr/bin/env python

import sys, os
import numpy as np
#sys.path.append('biceps/')
from edit_toolbox import *
import matplotlib.pyplot as plt


### Let's read the trajectory file ###
traj = np.load('traj_lambda1.00.npz')['arr_0'].item()
rest_type = traj['rest_type']
#rest_type = get_rest_type(traj)  # get restraint used in sampling
print("restraint_type", rest_type)

### Let's get the sampled parameters ###
#allowed_parameters = get_allowed_parameters(traj,rest_type=rest_type)
allowed_parameters = traj['allowed_parameters']
sampled_parameters = get_sampled_parameters(traj,rest_type=rest_type,allowed_parameters=allowed_parameters)
print("sampled parameters of ",rest_type[0],"for first 100 steps", sampled_parameters[0][:100])

### Let's plot the traces of sampled parameters ###
total_steps = len(sampled_parameters[0])
x = np.arange(1,total_steps+0.1,1)
fname = "traj_traces.png" #traj_traces_%s.pdf"%(rest_type[i])
plt.figure(figsize=(10,15))
n_rest = len(rest_type)
labels = []
for i in range(n_rest):
    if rest_type[i].count('_') == 0:
        if rest_type[i] == 'gamma':
            labels.append('$\%s$'%rest_type[i])
        else:
            labels.append('$%s$'%rest_type[i])
    elif rest_type[i].count('_') == 1:
        labels.append("$\%s_{%s}$"%(rest_type[i].split('_')[0],rest_type[i].split('_')[1]))
    elif rest_type[i].count('_') == 2:
        labels.append("$\%s_{{%s}_{%s}}$"%(rest_type[i].split('_')[0],rest_type[i].split('_')[1],rest_type[i].split('_')[2]))


for i in range(n_rest):
    plt.subplot(n_rest,1,i+1)
    plt.plot(x,sampled_parameters[i],label=labels[i])
    plt.ylabel(labels[i])
    plt.xlabel('steps')
    plt.legend(loc='best')
plt.tight_layout()
plt.savefig(fname)



### Let's plot the autocorrelation curve ###
#trace_labels = []
#for i in rest_type:
#    if i == 'gamma':
#        trace_labels.append('$\\gamma$')
#    else:
#        trace_labels.append('$\\sigma_{%s}$'%i)
max_tau=10000
autocorrs = []
for timeseries in sampled_parameters:
    autocorrs.append( g(np.array(timeseries), max_tau=max_tau) )

fname = "autocorrelation_curve.png"
plt.figure( figsize=(10,10))
for i in range(len(autocorrs)):
    plt.subplot(len(autocorrs),2,i+1)
    plt.plot(np.arange(max_tau+1),autocorrs[i])
    plt.ylabel('$g(\\tau)$ for %s'%labels[i])
    plt.xlabel('$\\tau$')
plt.tight_layout()
plt.savefig(fname)


### Let's fit the curve line with exponential decay and plot them ###
fname = "autocorrelation_curve_with_exp_fitting.png"
plt.figure( figsize=(10,10))
popts = []
for i in range(len(autocorrs)):
    yFit,popt = exponential_fit(autocorrs[i])
    plt.subplot(len(autocorrs),2,i+1)
    plt.plot(np.arange(max_tau+1), autocorrs[i])
    plt.plot(np.arange(max_tau+1), yFit, 'r--')
    plt.xlabel('$\\tau$')
    plt.ylabel('$g(\\tau)$ for %s'%labels[i]) 
    popts.append(popt)
plt.tight_layout()
plt.savefig(fname)

TAU = np.max(popts)
np.savetxt("popts.dat",popts)
print("Maximum tau = %s"%TAU)
# Apparently $\sigma_J$ decorrelates more slowly than the other two parameters. For the sake of this example, let's just use step = 1000 as $\tau$.

### Let's subsample the raw trajectory and compute JSDs ###
tau = int(1+2*TAU)
np.savetxt("tau.dat",[tau])
#traj_sec = np.load('results_ref_normal/traj_lambda1.00.npz')['arr_0'].item()  # load "trajectory" section
T_new = traj['trajectory'][::tau]    # subsample the raw trajectory based on tau
#np.save('subsample.npy',np.array(T_new,dtype=object))
#print T_new
#np.savez_compressed('subsample.npy',T_new)
#sys.exit()
fold = 10      # divide new subsampled trajectory into 10 folds
nsnaps = len(T_new)      # count number of snapshots of new subsampled trajectory
dx = int(nsnaps/fold)
#rounds = 100    # number of rounds for mixing data
#all_JSD=[]      # create JSD list
#all_JSDs=[[] for i in range(fold)]   # create JSD list of distribution
#traj = 'traj_lambda1.00.npz'
r_total = [[] for i in range(len(rest_type))]
r_max = [[] for i in range(len(rest_type))]
for subset in range(fold):
#    half = dx * (subset+1)/2
#    T1 = T_new[:half]     # first half of the trajectory
#    T2 = T_new[half:dx*(subset+1)]    # second half of the trajectory
    T_total = T_new[dx*subset:dx*(subset+1)]     # total trajectory
#    all_JSD.append(compute_JSD(T1,T2,T_total,rest_type,allowed_parameters))   # compute JSD
#    for r in range(rounds):      # now let's mix this dataset
#        mT1 = np.random.choice(len(T_total),len(T_total)/2,replace=False)    # randomly pickup snapshots (index) as the first part
#        mT2 = np.delete(np.arange(0,len(T_total),1),mT1)           # take the rest (index) as the second part
#        temp_T1, temp_T2 = [],[]
#        for snapshot in mT1:
#                temp_T1.append(T_total[snapshot])      # take the first part dataset from the trajectory
#        for snapshot in mT2:
#                temp_T2.append(T_total[snapshot])      # take the second part dataset from the trajectory
#        all_JSDs[subset].append(compute_JSD(temp_T1,temp_T2,T_total,rest_type,allowed_parameters))    # compute JSD
#np.save("all_JSD.npy", all_JSD)
#np.save("all_JSDs.npy", all_JSDs)
    for j in range(len(rest_type)):
        r_grid = np.zeros(len(allowed_parameters[j]))
        for k in T_total:
            ind = np.concatenate(k[4])[j]
            r_grid[ind]+=1
        r_total[j].append(r_grid)
        r_max[j].append(allowed_parameters[j][np.argmax(r_grid)])

fname = "block_avg.png"
plt.figure(figsize=(10,5*n_rest))
x=np.arange(1.,11.,1.)
colors=['red', 'blue','black','green']
for i in range(n_rest):
    total_max = allowed_parameters[i][np.argmax(traj['sampled_parameters'][i])]
    plt.subplot(n_rest,1,i+1)
    plt.plot(x,r_max[i],'o-',color=colors[i],label=labels[i])
    plt.xlabel('block')
    plt.ylabel('allowed '+labels[i])
    plt.ylim(min(allowed_parameters[i]),max(allowed_parameters[i]))
    plt.plot(9.8,total_max,'*',ms=20,color='green',label='total max')
    plt.legend(loc='best')
plt.tight_layout()
plt.savefig(fname)


print("Done")
sys.exit()
# compute_JSD function will compute JSD for all restraints
# all_JSD.shape = (fold,n_rest)
# all_JSDs.shape = (fold,round,n_rest)




### Let's plot JSD distribution ###

colors=['red', 'blue','black','green']

# convert shape of all_JSD from (fold,n_rest) to (n_rest,fold)
new_JSD = [[] for i in range(n_rest)]
for i in range(len(all_JSD)):
    for j in range(n_rest):
        new_JSD[j].append(all_JSD[i][j])

# compute mean, std of JSDs from each fold dataset of each restraint
JSD_dist = [[] for i in range(n_rest)]
JSD_std = [[] for i in range(n_rest)]
for rest in range(n_rest):
    for f in range(fold):
        temp_JSD = []
        for r in range(rounds):
            temp_JSD.append(all_JSDs[f][r][rest])
        JSD_dist[rest].append(np.mean(temp_JSD))
        JSD_std[rest].append(np.std(temp_JSD))
#print(JSD_dist     # JSD_dist.shape = (n_rest,fold))
#print(JSD_std      # JSD_std.shape = (n_rest,fold))
fname = "JSD_distribution.png"
plt.figure(figsize=(10,5*n_rest))
x=np.arange(10.,101.,10.)   # the dataset was divided into ten folds (this is the only hard coded part)
for i in range(n_rest):
    plt.subplot(n_rest,1,i+1)
    plt.plot(x,new_JSD[i],'o-',color=colors[i],label=labels[i])
    plt.hold(True)
    plt.plot(x,JSD_dist[i],'*',color=colors[i],label=labels[i])
    plt.fill_between(x,np.array(JSD_dist[i])+np.array(JSD_std[i]),np.array(JSD_dist[i])-np.array(JSD_std[i]),color=colors[i],alpha=0.2)
    plt.xlabel('dataset (%)')
    plt.ylabel('JSD')
    plt.legend(loc='best')
plt.tight_layout()
plt.savefig(fname)






