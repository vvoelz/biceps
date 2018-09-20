import sys, os
os.chdir('0')
for i in range(1,10):
	os.system('mkdir ../%d'%i)
	os.system('cp -r * ../%d'%i)
	
