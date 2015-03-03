import os, sys

if not os.path.exists('Gens'):
    os.mkdir('Gens')

for i in range(100):
    cmd = 'mdconvert Gens100.lh5 -t frame0.pdb -i %d -o Gens/Gen%d.pdb'%(i,i)
    print '>>', cmd
    os.system(cmd)

