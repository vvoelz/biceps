#!/bin/sh
#PBS -l walltime=4:00:00
#PBS -N cineromycin_B
#PBS -q normal 
#PBS -l nodes=1:ppn=20
#PBS -o cineromycin_B
#PBS 

cd $PBS_O_WORKDIR
python runme.py 
