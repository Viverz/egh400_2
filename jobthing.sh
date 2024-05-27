#!/bin/bash -l
#PBS -N job21
#PBS -l walltime=23:59:00
#PBS -l mem=100000mb
#PBS -l ncpus=8
#PBS -l ngpus=1
#PBS -l gputype=A100

module load python/3.11.5-gcccore-13.2.0

source /home/n9560751/home/venv31seg2/bin/activate

cd /home/n9560751/home/venv31seg2

python testsegment.py
