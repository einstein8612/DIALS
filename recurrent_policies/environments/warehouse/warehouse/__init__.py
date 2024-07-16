from gym.envs.registration import register
register(
    id='mini-warehouse-v0',
    entry_point='warehouse.envs:MiniWarehouse',
)
register(
    id='mini-warehouse-memory-v0',
    entry_point='warehouse.envs:MiniWarehouseMemory',
)
register(
    id='mini-warehouse-mini-v0',
    entry_point='warehouse.envs:MiniWarehouseMini',
)