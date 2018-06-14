import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers


# Constant
spatial_arguments = ['screen', 'minimap', 'screen2']

# Helper function used to normlize weights
def normalized_columns_initializer(std=1.0):
	def _initializer(shape, dtype=None, partition_info=None):
		out = np.random.randn(*shape).astype(np.float32)
		out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
		return tf.constant(out)
	return _initializer


class Network():
	def __init__(self, scope, action_spec, observation_spec):
		with tf.variable_scope(scope):

			# Number of non-spatial input data
			nonspatial_size = 12 # Number of simple data
			# Number of matrixs describing the minimap
			minimap_channels = 7
			# Number of matrixs describing the screen
			screen_channels = 17

			self.inputs_nonspatial = tf.placeholder(shape=[None,nonspatial_size], dtype=tf.float32)
			self.inputs_spatial_minimap = tf.placeholder(shape=[None,64,64,minimap_channels], dtype=tf.float32)  # Input layers
			self.inputs_spatial_screen = tf.placeholder(shape=[None,64,64,screen_channels], dtype=tf.float32)
			
			# Two convolution layers for screen input 
			self.screen_conv1 = tf.layers.conv2d(inputs=self.inputs_spatial_screen, filters=16, kernel_size=[5,5], strides=[1,1], padding='same', activation=tf.nn.relu)
			self.screen_conv2 = tf.layers.conv2d(inputs=self.screen_conv1, filters=32, kernel_size=[3,3], strides=[1,1], padding='same', activation=tf.nn.relu)

			# Two convolution layers for minimap input 
			self.minimap_conv1 = tf.layers.conv2d(inputs=self.inputs_spatial_minimap, filters=16, kernel_size=[5,5], strides=[1,1], padding='same', activation=tf.nn.relu)
			self.minimap_conv2 = tf.layers.conv2d(inputs=self.minimap_conv1, filters=32, kernel_size=[3,3], strides=[1,1], padding='same', activation=tf.nn.relu)

			# Number of output elements from the 2 convolutional neural networks
			screen_output_length = np.prod(self.screen_conv2.get_shape().as_list()[1:])
			minimap_output_length = np.prod(self.minimap_conv2.get_shape().as_list()[1:])

			# non_spatial layer
			self.state_representation = tf.concat([layers.flatten(self.screen_conv2), layers.flatten(self.minimap_conv2), self.inputs_nonspatial], axis=1)
			self.latent_vector_nonspatial = tf.layers.dense(inputs=self.state_representation, units=256, activation=tf.tanh)
			
			# non spatial policy
			self.non_spatial_policy = tf.layers.dense(inputs=self.latent_vector_nonspatial, units=len(action_spec.functions), activation=tf.nn.softmax, kernel_initializer=normalized_columns_initializer(0.01))

			# spatial policy			            
			self.latent_vector_spatial = tf.layers.conv2d(inputs=tf.concat([self.screen_conv2, self.minimap_conv2], axis=3), filters=1, kernel_size=[1,1], strides=[1,1], padding='same', activation=None)
			self.spatial_policy = tf.nn.softmax(tf.reshape(self.latent_vector_spatial, shape=[-1, 64 * 64]))

			# value
			self.value = tf.layers.dense(inputs=self.latent_vector_nonspatial, units=1, kernel_initializer=normalized_columns_initializer(1.0))

			# Update of global weights
			if scope != 'global':
				# Placeholder
				self.actions_base = tf.placeholder(shape=[None], dtype=tf.int32)
				self.actions_onehot_base = tf.one_hot(self.actions_base, 524, dtype=tf.float32)
				self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
				self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
				self.actions_arg_spatial = tf.placeholder(shape=[None],dtype=tf.int32)
				self.actions_onehot_arg_spatial = tf.one_hot(self.actions_arg_spatial, 64 * 64,dtype=tf.float32)

				# Compute the one-hot encoded action and its arguments
				self.responsible_outputs_base = tf.reduce_sum(self.non_spatial_policy * self.actions_onehot_base, [1])
				self.responsible_outputs_arg_spatial = tf.reduce_sum(self.spatial_policy * self.actions_onehot_arg_spatial, [1])

				# Loss functions
				self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
				self.log_non_spatial_policy = tf.log(tf.clip_by_value(self.non_spatial_policy, 1e-20, 1.0))

				# Compute the entropy
				self.entropy_base = - tf.reduce_sum(self.non_spatial_policy * self.log_non_spatial_policy)
				self.entropy_arg_spatial = - tf.reduce_sum(self.spatial_policy * tf.log(tf.clip_by_value(self.spatial_policy, 1e-20, 1.)))
				self.entropy = self.entropy_base
				self.entropy += self.entropy_arg_spatial

				# Compute loss for action
				self.policy_loss_base = - tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_outputs_base, 1e-20, 1.0)) * self.advantages)

				# Compute loss for action arguments
				self.policy_loss_arg_spatial = - tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_outputs_arg_spatial, 1e-20, 1.0))*self.advantages)

				# Compute overall loss
				self.policy_loss = self.policy_loss_base
				self.policy_loss += self.policy_loss_arg_spatial
				self.loss = self.value_loss + self.policy_loss - self.entropy * 0.001

				# Get gradients from local network using local losses
				local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
				self.gradients = tf.gradients(self.loss,local_vars)
				self.var_norms = tf.global_norm(local_vars)
				grads, self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
				
				# Apply local gradients to global network
				global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
				self.apply_grads = tf.train.AdamOptimizer(learning_rate=3e-5).apply_gradients(zip(grads, global_vars))