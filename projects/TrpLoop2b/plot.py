import os ,sys
import numpy as np
from matplotlib import pyplot as plt
a=np.arange(1,2569,1)
b=np.load('gamme.npy')
c=np.load('sigma.npy')
#print len(b)
#print len(c)
#sys.exit()
plt.figure()
plt.plot(a,b)
plt.ylim([0.2,5.0])
plt.show()
plt.plot(a,c)
plt.ylim([1,120])
plt.show()
