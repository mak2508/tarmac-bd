#!/bin/sh

DATE=$(date +%m_%d)

# torch.distributed config
HOST_PORT=100.97.16.141:9218
WORLD_SIZE=8

RANK=7
SEED=107

sbatch --job-name dist-test -p priority --gres gpu:1 --cpus-per-task 16 \
    --nodes 1 --ntasks-per-node 1 \
    --output logs/$DATE/sbatch.%j.out \
    --wrap "srun --output logs/$DATE/train.log.node%t.%j --error logs/$DATE/train.stderr.node%t.%j \
    python main.py \
        --overfit \
        --distributed \
        --distributed-rank $RANK \
        --distributed-world-size $WORLD_SIZE \
        --distributed-init-method 'tcp://$HOST_PORT' \
        --seed $SEED"
