#!/bin/bash
#$ -cwd
#$ -l m_mem_free=16G
#$ -pe threads 16
#$ -o output
#$ -e log
#$ -N chain8_sample5000
python inference_one_step.py -n 10000 -c 16 -w 100000
