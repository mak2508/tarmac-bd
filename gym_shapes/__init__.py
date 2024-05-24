from gym.envs.registration import register

register(
    id='ShapesEnv-v0',
    entry_point='gym_shapes.shapes_env:ShapesEnv',
)

register(
    id='SimpleShapesEnv-v0',
    entry_point='gym_shapes.simple_shapes_env:SimpleShapesEnv',
)