#!/bin/bash
#SBATCH -p learnfair
#SBATCH --gres gpu:1
#SBATCH -J marl
#SBATCH -o logs/07_25/slurm/4x4,c400,n4,s2,l70,att,fs-no,fi-no.%J.out
#SBATCH -e logs/07_25/slurm/4x4,c400,n4,s2,l70,att,fs-no,fi-no.%J.err

source .env/bin/activate
python main.py \
    --data-dir 'shapes_4x4_single_red' \
    --task 'colors.4,0,0' \
    --num-agents 4 \
    --step-size 2 \
    --num-steps 70 \
    --obs-height 5 \
    --obs-width 5 \
    --comm-mode 'from_states_rec_att' \
    --num-processes 16 \
    # --log
