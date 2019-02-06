#!/bin/sh
#PBS -l walltime=6:00:00
#PBS -N H030_biceps_100
#PBS -q highmem
##PBS -l mem=300gb
#PBS -l nodes=1:ppn=1
#PBS -o H030_biceps_100
#PBS

cd $PBS_O_WORKDIR

bash runme
