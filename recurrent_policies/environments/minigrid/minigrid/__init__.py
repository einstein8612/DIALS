from gym.envs.registration import register

register(
    id='MiniGrid-MemoryS17Random-modified-v0',
    entry_point='minigrid.envs:MemoryS17Random',
)
register(
    id='MiniGrid-MemoryS13Random-modified-v0',
    entry_point='minigrid.envs:MemoryS13Random',
)
register(
    id='MiniGrid-MemoryS13-modified-v0',
    entry_point='minigrid.envs:MemoryS13',
)

register(
    id='MiniGrid-MemoryS11-modified-v0',
    entry_point='minigrid.envs:MemoryS11',
)
