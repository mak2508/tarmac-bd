from gym.envs.registration import register

register(
    id='TrafficEnv-v0',
    entry_point='gym_traffic.traffic_env:TrafficEnv',
)
