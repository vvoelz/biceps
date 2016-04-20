import numpy as np
from matplotlib import pyplot as plt
import os, sys
import math

sim=np.loadtxt("NOE/rminus6_whole_state64.txt")
y1= sim**(-1./6.)
x=np.loadtxt("junk.txt")
y2=np.loadtxt("NOE/average_whole_state64.txt")
plt.figure(figsize=(8,8))
plt.scatter(x,y1,color="black",label="rminus6")
plt.scatter(x,y2,color="red",label="average")
plt.xlabel("experimental")
plt.ylabel("simulation")
plt.xlim([0,1.2])
plt.ylim([0,1.2])
plt.plot([0,1.2],[0,1.2],color="red")
plt.legend(loc='upper right')
plt.title('Comparison of exp vs sim (state64)')

z1=[]
for i in range(206):
                   a=(x[i]-y1[i])**2
                   z1.append(a)
print z1
b=0
for i in z1:
            b += i
print b
c=b/len(z1)
d=format(math.sqrt(c),'.3f')
print d
z2=[]
for i in range(206):
                   e=(x[i]-y2[i])**2
                   z2.append(e)
print z2
g=0
for i in z2:
            g += i
print g
h=g/len(z2)
j=format(math.sqrt(h),'.3f')
print j
#r1=np.corrcoef(x,y1)[0,1]
#r2=np.corrcoef(x,y2)[0,1]
plt.annotate('rms=%s'%d,xy=(0.8,0.1),color='black',fontsize=18)
plt.annotate('rms=%s'%j,xy=(0.8,0.2),color='red',fontsize=18)
#plt.annotate(r'$R^2=%d$'%r1,xy=(1.0,0.4),color='red',fontsize=22)
#plt.annotate(r'$R^2=0.319$',xy=(5.0,3.8),color='blue',fontsize=22)
plt.savefig('64.pdf')
plt.show()
