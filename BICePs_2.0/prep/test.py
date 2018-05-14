import sys, os, glob
import re
from prep import *
#self,scheme=None,states=0.0,indices=None, exp_data=None, top=None, data_dir=None, out_dir=None):
#        scheme: {'noe','J','cs_H','cs_Ha','cs_N','cs_Ca','pf'}
#        states: number of states
#        indices: experimental observable index (*.txt file)
#        exp_data: experimental measuremnets (*.txt file)
#        top: topology file (*.gro, pdb, etc.)
#        data_dir: data directory (should have *txt file inside)
#        out_dir: output directory
path='cs_H/cs/H/*txt'
states=50
indices='cs_H/cs_indices_NH.txt'
exp_data='cs_H/chemical_shift_NH.txt'
top='cs_H/8690.pdb'
data_dir=path
out_dir='test_cs_H'

#convert = lambda txt: int(txt) if txt.isdigit() else txt
#data = sorted(glob.glob(path),key=lambda x: [convert(s) for s in re.split("([0-9]+)",x)])
#print data
p=prep('cs_H',states=states,indices=indices,exp_data=exp_data,top=top,data_dir=data_dir,out_dir=out_dir)
p.write()
