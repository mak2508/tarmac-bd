# Multi-Agent RL + Communication

## Setup

### Set up `virtualenv`

```
git submodule init
git submodule update
conda create -n tarmac --python=3.6.6
conda activate tarmac
pip install -r requirements.txt
```
### Environments

#### SHAPES (from [Andreas et al., CVPR 2016][nmn])

Download data from [here](https://www.dropbox.com/s/cabudxh0f23nduf/08_12_shapes.zip), and place in 
`gym_shapes/data/shapes/` (for example `gym_shapes/data/shapes/data/shapes_3x3_single_red`, etc).

#### Traffic junction (from [Sukhbaatar et al., NIPS 2016][commnets])

#### House3D / MINOS

[TODO]

## Experiments

### SHAPES

#### Overfitting with 4 agents finding red, spawned at corners of a fixed image

`python main.py --env-name ShapesEnv-v0 --overfit --step-size 2 --num-steps 30`

This is equivalent to calling `python main.py --fix-spawn --fix-image`.

- `fix-spawn`: spawns agents at the corner of the image
- `fix-image`: fixes the image to `id = 25` in `gym_shapes/data/shapes/shapes_06_05_08:34/train.tiny.input.npy`
- `step-size`: grid coordinates to move with one agent step
- `num-steps`: max episode length

#### 4 agents + attentional 1-hop communication + different goals on a 30x30 image

`python main.py --env-name ShapesEnv-v0 --num-agents 4 --task colors.2,1,1 --data-dir 'shapes_3x3_single_red'  --comm-mode 'from_states_rec_att' --comm-num-hops 1`

![12](https://i.imgur.com/cBRqTLj.gif)
![13](https://i.imgur.com/B3a7iVq.gif)

### Traffic Junction

#### Difficulty=hard + 1-hop attentional communication

`python main.py --env-name TrafficEnv-v0 --difficulty hard --comm-mode from_states_rec_att --comm-num-hops 1`

~20% success rate; being tested at a harder car add rate of 0.50

![1](https://i.imgur.com/eZYBofC.gif)

#### Difficulty=hard + 2-hop attentional communication

`python main.py --env-name TrafficEnv-v0 --difficulty hard --comm-mode from_states_rec_att --comm-num-hops 2`

~92% success rate; being tested at a harder car add rate of 0.50

![2](https://i.imgur.com/OUfgd92.gif)

## Acknowledgements

- [pytorch-a2c-ppo-acktr][pytorch-a2c-ppo-acktr] by Ilya Kostrikov

[pytorch-a2c-ppo-acktr]: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
[nmn]: https://arxiv.org/abs/1511.02799
[commnets]: https://arxiv.org/abs/1605.07736

## License

MIT
