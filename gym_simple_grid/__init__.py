from gym.envs.registration import register

register(
    id='SimpleGridEnv-v0',
    entry_point='gym_simple_grid.simple_grid_env:SimpleGridEnv',
)