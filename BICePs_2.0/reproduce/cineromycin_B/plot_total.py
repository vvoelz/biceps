import sys, os
for i in range(1,10):
    os.chdir('%d/'%i)
    os.system('python plot.py')
    os.chdir('../')
