#!/bin/sh
#PBS -l walltime=24:00:00
#PBS -N cineromycin_B
#PBS -q normal 
#PBS -l nodes=1:ppn=20
#PBS -o cineromycin_B
#PBS 

cd $PBS_O_WORKDIR
cd 0/
python runme.py &
cd ../
cd 1/
python runme.py &
cd ../
cd 2/
python runme.py &
cd ../
cd 3/
python runme.py &
cd ../
cd 4/
python runme.py &
cd ../
cd 5/
python runme.py &
cd ../
cd 6/
python runme.py &
cd ../
cd 7/
python runme.py &
cd ../
cd 8/
python runme.py &
cd ../
cd 9/
python runme.py &
cd ../

wait
