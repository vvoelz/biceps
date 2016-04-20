import numpy as np

c=[]
a=dict()
for i in range(250):
	a[i]=np.loadtxt('NOE/rminus6_whole_state%d.txt'%i)
	c.append(a[i])
print c	
b=np.reshape(c,(250,206))
#for i in c:
#b=np.vstack((a[i]))
#	b.append(a)
#	print a
print b
print len(b)
print b.shape
np.savetxt('new_rminus6_249.txt',b)
