from network import Network
import global_var

import tensorflow as tf
import numpy as np
from pysc2.env import sc2_env, environment
from pysc2.lib import actions
import scipy.signal
import tensorflow as tf

# Constants
GAMMA = 0.99
spatial_arguments = ['screen', 'minimap', 'screen2']

# Processes PySC2 observations
def process_observation(observation, action_spec, observation_spec):
	# Signal representing if the episode is over
	episode_done = observation.step_type == environment.StepType.LAST
	# Reward signal
	reward = observation.reward
	# Observation signal cotaining the features
	features = observation.observation

	# nonspatial features
	nonspatial_stack = []
	nonspatial_stack = np.log(features['player'].reshape(-1) + 1.)
	nonspatial_stack = np.concatenate((nonspatial_stack, features['game_loop'].reshape(-1)))
	nonspatial_stack = np.expand_dims(nonspatial_stack, axis=0)
	
	# spatial_minimap features
	minimap_stack = np.stack((features['minimap']), axis=2)
	minimap_stack = np.expand_dims(minimap_stack, axis=0)
	
	# spatial_screen features
	screen_stack = np.stack((features['screen']), axis=2)
	screen_stack = np.expand_dims(screen_stack, axis=0)

	return reward, nonspatial_stack, minimap_stack, screen_stack, episode_done

# used to discount reward
def discount(x):
	return scipy.signal.lfilter([1], [1, -GAMMA], x[::-1], axis=0)[::-1]

# copy the global network weights into the local one
def copy_the_global_network(global_network, worker_network):
	global_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, global_network)
	worker_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, worker_network)
	op_holder = []
	for global_weight, worker_weight in zip(global_weights, worker_weights):
		op_holder.append(worker_weight.assign(global_weight))
	return op_holder

# Sample action from distrubution
def sample_from_distribution(distribution):
	sample = np.random.choice(distribution[0],p=distribution[0])
	sample = np.argmax(distribution == sample)
	return sample


class Worker():
	def __init__(self, worker_number, result_path, model_path, global_episodes, global_steps, map_name, action_spec, observation_spec):
		# Worker name
		self.name = "worker_" + str(worker_number)
		# Name of model path 										
		self.model_path = model_path
		# Name of result path
		self.result_path = result_path
		# The workers own result file
		self.result_file = open(self.result_path + "/"+ map_name +"_{}_results.csv".format(worker_number), 'w')

		# Global variable keeping track of the number of episodes done
		self.global_episodes = global_episodes
		# Global variable keeping track of the number of steps executed
		self.global_steps = global_steps
		# Function to add one to the global_episode counter
		self.increment_global_episodes = self.global_episodes.assign_add(1)
		# Function to add one to the global_steps counter
		self.increment_global_steps = self.global_steps.assign_add(1)

		# Variable contaitng information about the action specification
		self.action_spec = action_spec
		# Variable containg information about the obsevartion specification														
		self.observation_spec = observation_spec									

		# Create the local network and copy global weights to local network
		self.local_network = Network(self.name, action_spec, observation_spec)
		self.update_weights = copy_the_global_network('global', self.name)  
				
		# Setup the enviroment
		self.env = sc2_env.SC2Env(map_name=map_name)



		
	def train(self, episode_buffer, sess, value):
		episode_buffer = np.array(episode_buffer)
		obs_minimap = episode_buffer[:,0]
		obs_screen = episode_buffer[:,1]
		obs_nonspatial = episode_buffer[:,2]
		actions_base = episode_buffer[:,3]
		actions_args_spatial = episode_buffer[:,4]
		rewards = episode_buffer[:,5]
		values = episode_buffer[:,6]

		# Here we take the rewards and values from the episode_buffer, and use them to calculate the advantage and discounted returns.
		self.rewards_plus = np.asarray(rewards.tolist() + [value])
		self.value_plus = np.asarray(values.tolist() + [value])

		# Calculatin discounted_rewards and advantages
		discounted_rewards = discount(self.rewards_plus)[:-1]
		advantages = rewards + GAMMA * self.value_plus[1:] - self.value_plus[:-1]
		advantages = discount(advantages)

		# Update the global network using gradients from loss
		feed_dict = {self.local_network.target_v:discounted_rewards,
			self.local_network.inputs_spatial_screen:np.stack(obs_screen).reshape(-1,64,64,17),
			self.local_network.inputs_spatial_minimap:np.stack(obs_minimap).reshape(-1,64,64,7),
			self.local_network.inputs_nonspatial:np.stack(obs_nonspatial).reshape(-1,12),
			self.local_network.actions_base:actions_base,
			self.local_network.advantages:advantages,
			self.local_network.actions_arg_spatial:actions_args_spatial}
		_ = sess.run([self.local_network.apply_grads], feed_dict=feed_dict)

	


	def work(self,sess,coord,saver):

		# Start session to keep track of the number of global episodes done
		episode_count = sess.run(self.global_episodes)
		# Variable to keep track of steps this worker have done 
		total_steps = 0 							   
		
		with sess.as_default(), sess.graph.as_default():
			while True:
				#  Copy the weights from global network and put them into the worker network			 
				sess.run(self.update_weights)

				episode_buffer = [] 			# 
				episode_values = []				#
				episode_frames = []				# MEMORY
				episode_reward = 0				#
				episode_step_count = 0			#
				
				# Get first observation in a epsiode
				obs = self.env.reset()
				# Append observation to memory
				episode_frames.append(obs[0])
				# Process the observation, to split up the data
				reward, nonspatial_stack, minimap_stack, screen_stack, episode_done = process_observation(obs[0], self.action_spec, self.observation_spec)

				# minimap, screen and nonspatial features for first obsersvation in a episode
				s_minimap = minimap_stack
				s_screen = screen_stack
				s_nonspatial = nonspatial_stack
				
				while not episode_done:
					# Take an action using the policy network.
					non_spatial_policy_dist, spatial_policy_dist, value = sess.run([
						self.local_network.non_spatial_policy, 
						self.local_network.spatial_policy,
						self.local_network.value],
						feed_dict={
							self.local_network.inputs_spatial_minimap: minimap_stack,
							self.local_network.inputs_spatial_screen: screen_stack,
							self.local_network.inputs_nonspatial: nonspatial_stack})

					# Remove unavailable actions from the output distribution and then renormalize the posible actions
					non_spatial_policy_dist[0] += 1e-20
					for action_id, _ in enumerate(non_spatial_policy_dist[0]):
						if action_id not in obs[0].observation['available_actions']:
							non_spatial_policy_dist[0][action_id] = 0.
					non_spatial_policy_dist[0] /= np.sum(non_spatial_policy_dist[0])
					
					# Sample an an anction
					action_id = sample_from_distribution(non_spatial_policy_dist) 
					
					# Sample spatial action (aka the input to the action)
					arg_sample_spatial_abs = sample_from_distribution(spatial_policy_dist)
					arg_sample_spatial = [arg_sample_spatial_abs % 64, arg_sample_spatial_abs / 64]

					# Insert the arguments of the action ID
					arguments = []
					for argument in self.action_spec.functions[action_id].args:
						if argument.name in spatial_arguments:
							argument_value = arg_sample_spatial
						else:
							argument_value = [0]
						arguments.append(argument_value)

					# Call function with an actions and its arguments, to produce an action object
					a = actions.FunctionCall(action_id, arguments)

					# Perform the action and get a new observation
					obs = self.env.step(actions=[a])

					# Process the observation
					r, nonspatial_stack, minimap_stack, screen_stack, episode_done = process_observation(obs[0], self.action_spec, self.observation_spec)

					# How to deal with the state, given information on wheter the epsiode is over or not
					if not episode_done:
						episode_frames.append(obs[0])
						s1_minimap = minimap_stack
						s1_screen = screen_stack
						s1_nonspatial = nonspatial_stack
					else:
						s1_minimap = s_minimap
						s1_screen = s_screen
						s1_nonspatial = s_nonspatial
					
					# Append latest state to buffer
					episode_buffer.append([s_minimap, s_screen, s_nonspatial,action_id,arg_sample_spatial_abs,r,value[0,0]])
					episode_values.append(value[0,0])

					# Increment episode reward with the reward gained from the previous action
					episode_reward += r
					# Update state variable
					s_minimap = s1_minimap
					s_screen = s1_screen
					s_nonspatial = s1_nonspatial
					# Increment global steps, total_steps and epsiode count with 1				 
					sess.run(self.increment_global_steps)
					total_steps += 1
					episode_step_count += 1
					
					# If the episode hasn't ended, and the experience buffer is full, make an update step using that experience episode_buffer.
					if len(episode_buffer) == 40 and not episode_done:
						# Estimate the value function which is used to compute the advantage
						value = sess.run(self.local_network.value, 
							feed_dict={self.local_network.inputs_spatial_minimap: minimap_stack, self.local_network.inputs_spatial_screen: screen_stack,self.local_network.inputs_nonspatial: nonspatial_stack})[0,0]

						# Train the global network using the information gained
						self.train(episode_buffer,sess,value)

						# Reset replay_memory
						episode_buffer = []
						# Update the local weights
						sess.run(self.update_weights)
					if episode_done:
						break
				
				# Increment episode counter										
				episode_count += 1

				# Get epsiode reward
				episode_reward = obs[0].observation['score_cumulative'][0]

				# If new best score set it
				if global_var._max_score < episode_reward:
					global_var._max_score = episode_reward

				# Compute running mean
				global_var._running_avg_score = (2.0 / 101) * (episode_reward - global_var._running_avg_score) + global_var._running_avg_score

				# Print stuff to screen
				print("{} Step #{} Episode #{} Reward: {}".format(self.name, total_steps, episode_count, episode_reward))
				print("Total Steps: {}\tTotal Episodes: {}\tMax Score: {}\tAvg Score: {}".format(sess.run(self.global_steps), sess.run(self.global_episodes), global_var._max_score, global_var._running_avg_score))

				# Write to the workers result file
				self.result_file.write("{} {} {} \n".format(episode_reward, sess.run(self.global_steps), sess.run(self.global_episodes)))

				# Update the network using the episode buffer at the end of the episode.
				if len(episode_buffer) != 0:
					self.train(episode_buffer,sess,0.)
				
				# Increment global episodes with 1
				sess.run(self.increment_global_episodes)
