from gym.envs.registration import register

register(
    id='custom_env-v0',                                   # Format should be xxx-v0, xxx-v1....
    entry_point='custom_gym.envs:CustomEnv',              # Expalined in envs/__init__.py
)