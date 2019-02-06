import os, sys
a=[8,9,10,11,12,13,14,15,16,17,18,19]
b=[7,8,9,10,11,12,13,14,15,16,17,18]
for i in range(len(a)):
	os.system('mv lattice_model_%d.noe lattice_model_%d.noe'%(a[i],b[i]))
#        os.system('mv error_%d.dat error_%d.dat'%(a[i],b[i]))
