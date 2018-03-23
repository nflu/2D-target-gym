from gym.envs.registration import register

register(
	id="target-v0",
	entry_point="gym_target.envs:TargetEnv",
)
register(
	id="target-dynamics-v0",
	entry_point="gym_target.envs.TargetDynamicsEnv",
)
