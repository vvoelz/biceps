import os, sys
import numpy as np
from matplotlib import pyplot as plt

# load rminus6 value (nm) from NOE folder for each state
a=dict()
for i in range(250):
	a[i]=[]
	a[i]=np.loadtxt('NOE/rminus6_whole_state%d.txt'%i)

# Append rminus6 value from each state for each atom pair
c=dict()
for j in range(190):
	c[j]=[]
	for i in range(250):
		c[j].append(a[i][j])

# load exp data (nm)
b=np.loadtxt('junk.txt')

#calculate sse for each atom pair
d=dict()
for i in range(190):
	d[i]=[]
	for k in c[i]:
		d[i].append(((k-b[i])**2.))
e=[]
for i in range(190):
	e.append(sum(d[i]))
#out=np.reshape(e,(206,1))
#print out
#calculate sse for each state
f=dict()
for i in range(250):
	f[i]=[]
	for j in range(190):
		f[i].append((b[j]-a[i][j])**2.)
g=[]
for i in range(250):
	g.append(sum(f[i]))	
#out1=np.reshape(g,(250,1))
#print out1
#sys.exit()

fig=plt.figure()
ax=fig.add_subplot(111)
width=0.35
x=np.arange(0,190,1)
rect=ax.bar(x,e,width,color='red')
#plt.plot(x,e)
ax.set_xlabel('atom pairs')
ax.set_ylabel('sse')
ax.set_xlim([0,191])
for i in range(len(e)):
	if e[i]>10:
		ax.text(x[i],e[i],str(i),color='g')
plt.show()
fig=plt.figure()
ax=fig.add_subplot(111)
width=0.35
x2=np.arange(0,250,1)
rect=ax.bar(x2,g,width,color='red')
#plt.plot(64,g[64],"*")
ax.set_xlabel('state number')
ax.set_ylabel('sse')
plt.show()

