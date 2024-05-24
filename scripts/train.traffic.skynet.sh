#!/bin/bash
#SBATCH -p long
#SBATCH --gres gpu:1
#SBATCH -J marl
#SBATCH -x walle
#SBATCH -o logs/09_20/slurm/traffic,hard,comm-att,comm-hops-2,comm-size-32.out
#SBATCH -e logs/09_20/slurm/traffic,hard,comm-att,comm-hops-2,comm-size-32.err

source .env/bin/activate
python main.py \
    --env-name 'TrafficEnv-v0' \
    --difficulty 'hard' \
    --comm-mode 'from_states_rec_att' \
    --comm-num-hops 2 \
    --comm-size 32 \
    --num-processes 32 \
    --log