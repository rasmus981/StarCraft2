from network import Network
from worker import Worker
import global_var

import sys
from absl import flags
import tensorflow as tf
import os
import threading
from time import sleep

from pysc2.env import sc2_env

FLAGS = flags.FLAGS 	# Stuff needed
FLAGS(sys.argv) 		# Stuff needed



# CONSTANTS

map_name = "CollectMineralShards"		# String indicating which map we want to play
load_model = False						# Boolean indicating if we want to load a model
model_path = './model' 					# Path to model
results_path= './results'				# Path to results
n_agents = 4							# Number of agents



# Start of program 

def main():
	# Action and observation space
	action_spec = sc2_env.SC2Env(map_name=map_name).action_spec()
	observation_spec = sc2_env.SC2Env(map_name=map_name).observation_spec()

	# Reset global default graph
	tf.reset_default_graph()

	# If ther don't exists a folder ./model, create it
	if not os.path.exists(model_path):
		os.makedirs(model_path)

	# If ther don't exists a folder ./reults, create it
	if not os.path.exists(results_path):
		os.makedirs(results_path)

	# Intialize global max_score and average_score
	global_var.initialize()

	# Variables to keep track of the global number of episodes and steps performed
	global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
	global_steps = tf.Variable(0, dtype=tf.int32, name='global_steps', trainable=False)

	# Create the global network for A3C
	global_network = Network('global', action_spec, observation_spec)

	# List of worker
	workers = []

	# Construt workers and add them to the workers list
	for i in range(n_agents):
		workers.append(Worker(i, results_path, model_path, global_episodes, global_steps, map_name, action_spec, observation_spec))

	# Construct a saver object used to saves and load model weights
	saver = tf.train.Saver(max_to_keep=5)


	# Create tensorflow session, which eiter load a model or initialize global variables
	with tf.Session() as sess:
		coord = tf.train.Coordinator()
		if load_model == True:
			ckpt = tf.train.get_checkpoint_state(model_path)
			saver.restore(sess, ckpt.model_checkpoint_path)
		else:
			sess.run(tf.global_variables_initializer())
			
		# List with worker threads
		worker_threads = []
		for worker in workers:
			# Set the worker threads to call worker.work when thread starts
			worker_work = lambda: worker.work(sess, coord, saver)
			t = threading.Thread(target=(worker_work))
			t.start()
			sleep(2.0)
			worker_threads.append(t)
		coord.join(worker_threads)

main()
