import os, sys
import numpy as np
from matplotlib import pyplot as plt

# load rminus6 value (nm) from NOE folder for each state
a=dict()
for i in range(250):
	a[i]=[]
	a[i]=np.loadtxt('NOE/rminus6_whole_state%d.txt'%i)*10.0

# Append rminus6 value from each state for each atom pair
c=dict()
for j in range(206):
	c[j]=[]
	for i in range(250):
		c[j].append(a[i][j])

# load exp data (nm)
b=np.loadtxt('junk.txt')*10.0

#calculate specific state sse

for j in range(206):
	s=[]
	for i in range(250):
		s.append((a[i][j]-b[j])**2.)
	print s.index(min(s)) 
sys.exit()
plt.figure()
width=1
x=np.arange(0,250,1)
plt.xlim([0,250])	
plt.bar(x,s,width,color='red')
plt.xlabel('state')
plt.show()
sys.exit()
#s225=[]
#s239=[]
#s0=[]
#for i in range(206):
#	s225.append((a[225][i]-b[i])**2.)
#	s239.append((a[239][i]-b[i])**2.)
#	s0.append((a[0][i]-b[i])**2.)
#sse=[]
#for i in range(206):
#	sse.append((s239[i]-s0[i]))	
#plt.figure()
#width=0.5
#x=np.arange(0,206,1)
#plt.xlim([0,207])
#plt.bar(x,sse,width,color='red')
#plt.xlabel('atom pair')
#plt.title('difference between s239 and s0')
#for i in range(len(sse)):
#               if abs(sse[i])>1:
#                      plt.text(x[i],sse[i],str(i),color='g')
#plt.savefig('sse_diff.pdf')
#plt.show()
#sys.exit()
#plt.figure()
#width=0.5
#x=np.arange(0,206,1)
#plt.xlim([0,207])
#plt.bar(x,s225,width,color='red')
#plt.xlabel('atom pair')
#plt.title('state 225')
#plt.savefig('state225.pdf')
#plt.show()
#plt.figure()
#plt.xlim([0,207])
#plt.bar(x,s239,width,color='black')
#plt.xlabel('atom pair')
#plt.title('state 239')
#plt.savefig('state239.pdf')
#plt.show()
#plt.figure()
#plt.xlim([0,207])
#plt.bar(x,s0,width,color='blue')
#plt.xlabel('atom pair')
#plt.title('state 0')
#plt.savefig('state0.pdf')
#plt.show()
#sys.exit()

#calculate sse for each atom pair
d=dict()
for i in range(206):
	d[i]=[]
	for k in c[i]:
		d[i].append(((k-b[i])**2.))
#plt.plot(x,e)
#ax.set_xlim([0,251])
#for j in range(len(e)):
#               if e[i]>20:
#                       ax.text(x[i],e[i],str(i),color='g
#for i in range(206):
#	plt.figure()
#	width=0.35
#	x=np.arange(0,250,1)
#	plt.bar(x,d[i],width,color='red')
#plt.plot(x,e)
#	plt.xlabel('state')
#	plt.plot(64,d[i][64],"*")
#	plt.ylabel('sse')
#	ax.set_xlim([0,251])
#for j in range(len(e)):
#        	if e[i]>20:
#                	ax.text(x[i],e[i],str(i),color='g')
#	plt.savefig('sse_atom_pair_%d.pdf'%i)
#	plt.show()
#sys.exit()

e=[]
for i in range(206):
	e.append(sum(d[i]))
#out=np.reshape(e,(206,1))
#print out
#calculate sse for each state
f=dict()
for i in range(250):
	f[i]=[]
	for j in range(206):
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
x=np.arange(0,206,1)
rect=ax.bar(x,e,width,color='red')
#plt.plot(x,e)
ax.set_xlabel('atom pairs')
ax.set_ylabel('sse')
ax.set_xlim([0,207])
for i in range(len(e)):
	if e[i]>20:
		ax.text(x[i],e[i],str(i),color='g')
plt.savefig('sse_atom_pair.pdf')
plt.show()
fig=plt.figure()
ax=fig.add_subplot(111)
width=0.35
x2=np.arange(0,250,1)
rect=ax.bar(x2,g,width,color='red')
plt.plot(64,g[64],"*")
ax.set_xlabel('state number')
ax.set_ylabel('sse')
plt.savefig('sse_state.pdf')
plt.show()

