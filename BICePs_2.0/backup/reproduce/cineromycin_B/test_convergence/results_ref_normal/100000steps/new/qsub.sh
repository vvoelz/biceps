#!/bin/sh
#PBS -l walltime=24:00:00
#PBS -N cal_ac_100000 
#PBS -q normal 
#PBS -l nodes=1:ppn=4
#PBS -o cal_ac_100000 
#PBS 

cd $PBS_O_WORKDIR

python compute_ac.py
