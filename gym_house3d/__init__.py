from gym.envs.registration import register

register(
    id='House3DEnv-v0',
    entry_point='gym_house3d.house3d_env:House3DEnv',
)
