import os, sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import mlab as mlab
import math
color=['black','red','green','yellow','cyan','blue','magenta','orange','indigo','pink','darkred','indianred','springgreen','purple','goldenrod','peru','y','lightskyblue','blueviolet','gold']

y=[1,2,3,4,5,6,7,8]

mu, sigma=0, 0.5
s = np.random.normal(mu, sigma, 8)
#x=np.linspace(-3,3,100)
#plt.plot(x,mlab.normpdf(x,mu,sigma))
#plt.show()
#sys.exit()
#y=dict()

#for i in range(20):
#	y[i]=np.loadtxt('error_%d.dat'%i)
plt.figure(figsize=(12,8))
for i in range(20):
        x=np.loadtxt('error_%d.dat'%i)

	plt.plot(x,y,'o-',color=color[i],label='RUN_%d'%i)
plt.ylabel('distance indices')
plt.xlabel('noise')
plt.ylim(0,9)
plt.title('noise distribution')
plt.legend(loc='upper right',fontsize=8)

plt.show()
