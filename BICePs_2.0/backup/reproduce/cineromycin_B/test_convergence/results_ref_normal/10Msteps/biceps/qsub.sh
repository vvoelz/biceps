#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -N new_JSD100 
#PBS -q normal 
#PBS -l nodes=1:ppn=4
#PBS -o new_JSD100 
#PBS 

cd $PBS_O_WORKDIR
python compute_JSD.py
