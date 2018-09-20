import sys, os
import numpy as np
from matplotlib import pyplot as plt

BS_total=[]
pop_total=[]
for i in range(10):
    BS=np.loadtxt('%d/BS.dat'%i)[2][0]
    pop=np.loadtxt('%d/populations.dat'%i)[:,2]
    BS_total.append(BS)
    pop_total.append(pop)
avg_BS=np.mean(BS_total,axis=0)
avg_pop=np.mean(pop_total,axis=0)
print 'avg_BS',avg_BS
l=np.argsort(avg_pop)[::-1]
for i in range(10):
    print "state",l[i], 'pop', avg_pop[l[i]]
