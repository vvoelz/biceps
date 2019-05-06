import sys, os
import numpy as np

JSD = np.load('all_JSD.npy')[-1]
JSDs = np.load('all_JSDs.npy')[-1]

for i in range(len(JSD)):
	
	diff = JSD[i]-JSDs[:,i]
	rank = sum(1 for number in diff if number > 0)
	print 'rank', rank
	print 'len(JSD)/20',len(JSDs)/20 
	print rank <= len(JSDs)/20
		

