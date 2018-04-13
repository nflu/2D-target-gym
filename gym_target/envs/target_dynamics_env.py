import numpy as np
import gym 
from gym import spaces

class TargetDynamicsEnv(gym.Env):

	def __init__(self):
		print("EEE")
		
		# defines size of environment, dimension, and time step size
		self.timeHorizon = 500
		boxSizeX = 500.0
		boxSizeY = 500.0
		self.timestep_size = 0.1
		maxLambda = np.sqrt((2*boxSizeX)**2 + (2*boxSizeY)**2) #maximum distance between any two points in the space
		maxVelocity = maxLambda / self.timestep_size #velocity to move the maximum distance in minimum time
		#maxAccel = maxVelocity / self.timestep_size #I think this is unreasonably high
		maxAccel = 10.0 #I'm temporarily limiting it to this to make training easier
		#update: this value doesn't have an effect unless I enforce it with clipping see https://github.com/openai/baselines/issues/121

		self.high = np.array([boxSizeX, boxSizeY, maxVelocity, maxVelocity, maxLambda, boxSizeX, boxSizeY, maxAccel, maxAccel])
		#note that this describes the limits on the observation space.
		#right now the observation space is the state vector concatenated with the target set point
		self.stateDim = 5
		#(x, y, x', y', lambda)
		#if the dimension is increased and the new dimensions will be used in the observation
		#the dimension of the box must also increase.
		self.action_space = spaces.Box(low = -self.high[-2:], high = self.high[-2:], dtype = np.float32) #xAccel and yAccel
		self.observation_space = spaces.Box(low = -self.high[:self.stateDim+2], high = self.high[:self.stateDim+2], dtype = np.float32)
		self.curr_step = 0

	def randomPoint(self):
		point = np.zeros(2)
		point[0] = np.random.randint(low = -self.high[0], high = self.high[0])
		point[1] = np.random.randint(low = -self.high[1], high = self.high[1])
		return point

	def randomState(self):
		point = self.randomPoint()
		return np.append(np.append(point, np.zeros(self.stateDim - 3)), self.target(point)) #start at stationary point

	def _get_reward(self):
		'''
		return -self.target(self.state[:2]) #1
		
		if self.curr_step >= self.timeHorizon: #2
			return -self.target(self.state[:2])
		return 0

		if self.curr_step >= self.timeHorizon: #3 
			return -self.state[-1] 			   #needs lambda to record min distance
		return 0
		'''
		
		return self.lastLambda - self.state[-1] #4,5 change lambda appropriately



	'''
	def _step(self, action):
		return step(self, action)
	'''

	'''
	def _reset(self):
		return reset(self)
	'''

	def step(self, action):
		self._take_action(action)
		self.curr_step += 1
		reward = self._get_reward()
		ob = np.append(self.state, self.targetPoint)
		episode_over = self.curr_step >= self.timeHorizon
		return ob , reward, episode_over, {}

	def reset(self):
		self.targetPoint = self.randomPoint()
		def l(point):
			return np.linalg.norm(point - self.targetPoint)

		self.target = l
		self.state = self.randomState()
		self.curr_step = 0
		self.lastLambda =  float("inf")
		self.viewer = None
		ob = np.append(self.state, self.targetPoint)
		return ob

	def _take_action(self, action):
		self.state[0] = self.state[0] + self.timestep_size * self.state[2]
		self.state[1] = self.state[1] + self.timestep_size * self.state[3]
		self.state[2] = self.state[2] + self.timestep_size * action[0]
		self.state[3] = self.state[3] + self.timestep_size * action[1]
		if abs(self.state[0]) > self.high[0]:
			self.state[2] = -self.state[2]
		if abs(self.state[1]) > self.high[1]:
			self.state[3] = -self.state[3]
		self.lastLambda = self.state[-1]
		#self.state[-1] = min(self.state[-1], self.target(self.state[:2]))
		self.state[-1] = self.target(self.state[:2])

	def _render(self, mode="human", close=False):
		# I have this problem currently https://github.com/openai/gym/issues/893
		# The fix is to use _close() before close() is called  
		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(int(2*self.high[0]), int(2*self.high[1]))
			self.viewer.set_bounds(-self.high[0], self.high[0], -self.high[1], self.high[1])
			target = rendering.make_circle(2)
			target.add_attr(rendering.Transform(translation=self.targetPoint))
			target.set_color(0,0,0)
			self.viewer.add_geom(target)
			agent = rendering.make_circle(2)
			self.agent_translation = rendering.Transform(translation=self.state[:2])
			agent.add_attr(self.agent_translation)
			agent.set_color(1,0,0)
			self.viewer.add_geom(agent)
		else:
			self.agent_translation.set_translation(self.state[0], self.state[1])

		return self.viewer.render(return_rgb_array = mode=='rgb_array')

	def _close(self):
		if self.viewer is not None:
			self.viewer.close()
			self.viewer = None

	def _seed(self):
		#TODO
		pass