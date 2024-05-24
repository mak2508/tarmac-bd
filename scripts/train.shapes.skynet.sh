#!/bin/bash
#SBATCH -p long
#SBATCH --gres gpu:1
#SBATCH -J marl
#SBATCH -w t1000
#SBATCH -o logs/07_05/slurm/10x10,c0,n8,s2,l200,att,fs-no,fi-no.out
#SBATCH -e logs/07_05/slurm/10x10,c0,n8,s2,l200,att,fs-no,fi-no.err

source .env/bin/activate
python main.py \
    --data-dir 'shapes_10x10_single_red' \
    --task 'colors.0' \
    --num-agents 8 \
    --step-size 2 \
    --num-steps 200 \
    --obs-height 5 \
    --obs-width 5 \
    --comm-mode 'from_states_rec_att' \
    --num-processes 32 \
    --log