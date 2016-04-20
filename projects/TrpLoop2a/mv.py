import os, sys
for i in range(250):
        os.system("mv NOE/txt/average_state%d.txt NOE/"%(i))
        os.system("mv NOE/txt/rminus6_state%d.txt NOE/"%(i))
        os.system("mv NOE/txt/average_whole_state%d.txt NOE/"%(i))
        os.system("mv NOE/txt/rminus6_whole_state%d.txt NOE/"%(i))
#        os.system("mv Gens/Gens/Gens%d.pdb Gens/"%(i))

