import sys, os
import numpy as np
a=np.loadtxt("populations.dat")
b=[]
for i in a:
#	if i == 1:
#		print 100000
#	else:
	        print -np.log((i/float(sum(a))))
#	for j in b:
#        	f=-np.log(j) # units in kT
#        	print f
