from gym.envs.registration import register
register(
    id='treemaze-v0',
    entry_point='treemaze.envs:TreeMazeEnv',
)