import sys, os
import numpy as np
from matplotlib import pyplot as plt


plt.figure( figsize=(3.3, 3) )
nblock=np.arange(1,11) # 10 blocks
sigma = [1.1,1.2,1.3,1.1,1.0,1.3,1.2,1.1,1.1,1.1]  # make up some sigma values
avg = [1.08]   # make up an average sigma values

plt.plot(nblock,sigma,'o-',color='black')
plt.plot([1,10],[avg[0],avg[0]],color='blue')
plt.ylim(0,4)
plt.savefig('block_avg.pdf')
plt.show()


