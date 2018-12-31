from gym.envs.registration import register

register(
    id='Bicycle-v0',
    entry_point='envs.BicycleBalanceEnv:BicycleBalanceEnv',
)

register(
    id='Bicycle-v1',
    entry_point='envs.BicycleGotoEnv:BicycleGotoEnv',
)
