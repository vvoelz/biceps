import sys, os
import numpy as np
from matplotlib import pyplot as plt

seq = [1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09,1.10,1.20,1.50,3.00,6.00,15.00]

for i in seq:
    print i
    JSD = np.load('d_%.2f/all_JSD.npy'%i)
    JSDs = np.load('d_%.2f/all_JSDs.npy'%i)

    for k in range(len(JSD[0])):
        plt.figure(figsize=(10,5))
        for j in range(len(JSD)):
            plt.subplot(2,5,j+1)
            counts,bins = np.histogram(JSDs[j][:,k],bins = np.arange(min(JSDs[j][:,k]),max(JSDs[j][:,k]),0.00001))
            plt.step(bins[0:-1],counts,'black',label = '$P_{JSD}$')
           # print JSD[j][0]
            plt.plot(JSD[j][k],0.0,'*',ms=10,color='red')
            plt.yticks([])
            plt.xticks(fontsize=6)
            plt.legend(loc='best',fontsize=8)
            plt.title('%d'%(10*(j+1))+'%',fontsize=10)
        plt.tight_layout()
        plt.savefig('d_%.2f/JSD_%d.pdf'%(i,k))
        plt.close()



