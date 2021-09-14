#!/bin/bash
#$ -cwd
#$ -l m_mem_free=16G
#$ -pe threads 4
#$ -o output
#$ -e log
#$ -N chain16_sample10000
python inference_one_step_fixed_noise.py -n 5000 -c 4 -w 1000
