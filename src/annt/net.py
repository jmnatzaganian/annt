# net.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Date Created   : 03/30/15
#	
# Description    : Module for various artificial neural networks.
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
Module for various artificial neural networks.

G{packagetree annt}
"""

__docformat__ = 'epytext'

# Native imports
from abc import ABCMeta, abstractmethod

# Third party imports
import numpy as np

# Program imports
from annt.activation import create_activation
from annt.timers     import MultiTimer, pretty_time

###############################################################################
########## Class Templates
###############################################################################

class Net(object):
	"""
	Base class description for a network.
	"""
	__metaclass__ = ABCMeta
	
	@abstractmethod
	def __init__(self):
		"""
		Initialize the class instance.
		"""
	
	@abstractmethod
	def step(self, x, y=None):
		"""
		Compute a single step of the network.
		
		@param x: The input data to compute for this step.
		
		@param y: The expected output.
		"""
	
	@abstractmethod
	def run(self, train_data, train_labels, test_data, test_labels,
		nepochs=1):
		"""
		Simulate the entire network.
		
		@param train_data: The data to train with. This must be an iterable
		returning a numpy array.
		
		@param train_labels: The training labels. This must be an iterable
		with the same length as train_data.
		
		@param test_data: The data to test with. This must be an iterable
		returning a numpy array.
		
		@param test_labels: The testing labels. This must be an iterable
		with the same length as train_data.
		
		@param nepochs: The number of training epochs to perform.
		
		@return: A tuple containing the training and test costs/accuracies.
		"""
	
	@abstractmethod
	def initialize_weights(self, shape, min_weight=-1, max_weight=1):
		"""
		Initialize the weights of the network. Initialization is done randomly.
		
		@param shape: The number of nodes in the entire network, excluding any
		bias terms. This parameter must be a sequence.
		
		@param min_weight: The minimum weight value.
		
		@param max_weight: The maximum weight value.
		"""
	
	def _run(self, x, y=None):
		"""
		Execute the network for this batch of data
		
		@param x: The input data to compute for this step.
		
		@param y: The expected output.
		
		@return: The expected outputs for each input.
		"""
		
		if y is None:
			return np.array(map(self.step, x))
		else:
			return np.array(map(self.step, x, y))
	
	def enable_learning(self):
		"""
		Enables learning for the network.
		"""
		
		self.learning = True
	
	def disable_learning(self):
		"""
		Disables learning for the network.
		"""
		
		self.learning = False

class SupervisedCostNet(Net):
	"""
	Base class description for network that computes a cost function using
	unsupervised training.
	"""
	__metaclass__ = ABCMeta
	
	@abstractmethod
	def cost(self, y, y_exp):
		"""
		Compute the cost function
		
		@param y: The true output.
		
		@param y_exp: The expected output.
		
		@return: The cost.
		"""
	
	def _print_train(self, epoch, nepochs, train_cost):
		"""
		Print the details about training.
		
		@param epoch: The current epoch number.
		
		@param nepochs: The total number of epochs.
		
		@param train_cost: A numpy array containing the training cost at each
		epoch.
		"""
		
		print '\nEpoch {0} of {1}:'.format(epoch + 1, nepochs)
		print '  Training Cost     : {0}'.format(train_cost[epoch])
		print '  Training Time     : {0}'.format(
			self.timers.get_elapsed_time('train_epoch', True))
	
	def _print_test(self, epoch, test_cost):
		"""
		Print the details about testing.
		
		@param epoch: The current epoch number.
		
		@param test_cost: A numpy array containing the testing cost at each
		epoch.
		"""
		
		print '  Testing Cost      : {0}'.format(test_cost[epoch])
		print '  Testing Time      : {0}'.format(
			self.timers.get_elapsed_time('test_epoch', True))
	
	def _print_final(self, nepochs, train_cost, test_cost):
		"""
		Print the details final details.
		
		@param nepochs: The total number of epochs.
		
		@param train_cost: A numpy array containing the training cost at each
		epoch.
		
		@param test_cost: A numpy array containing the testing cost at each
		epoch.
		"""
		
		print '\n' + '*' * 79
		print '\nBest Training Cost : {0} at Epoch {1}'.format(np.min(
			train_cost), np.argmin(train_cost) + 1)
		print 'Best Testing Cost  : {0} at Epoch {1}'.format(np.min(
			test_cost), np.argmin(test_cost) + 1)
		print '\nTotal Execution Time        : {0}'.format(
			self.timers.get_elapsed_time('global', True))
		print 'Total Training Time         : {0}'.format(
			self.timers.get_elapsed_time('train', True))
		print 'Average Training Epoch Time : {0}'.format(
			pretty_time(self.timers.get_elapsed_time('train') / nepochs))
		print 'Total Testing Time          : {0}'.format(
			self.timers.get_elapsed_time('test', True))
		print 'Average Testing Epoch Time  : {0}'.format(
			pretty_time(self.timers.get_elapsed_time('test') / nepochs))
	
	def run(self, train_data, train_labels, test_data, test_labels,
		nepochs=1, verbose=True):
		"""
		Simulate the entire network.
		
		@param train_data: The data to train with. This must be an iterable
		returning a numpy array.
		
		@param train_labels: The training labels. This must be an iterable
		with the same length as train_data.
		
		@param test_data: The data to test with. This must be an iterable
		returning a numpy array.
		
		@param test_labels: The testing labels. This must be an iterable
		with the same length as train_data.
		
		@param nepochs: The number of training epochs to perform.
		
		@param verbose: If True, details will be printed after each epoch.
		
		@return: A tuple containing the training and test costs.
		"""
		
		# Make some timers
		self.timers = MultiTimer()
		self.timers.add_timers('global', 'train', 'train_epoch', 'test',
			'test_epoch')
		self.timers.stop_timers('train', 'train_epoch', 'test', 'test_epoch')
		
		_run = self._run; cost = self.cost
		train_cost = np.zeros(nepochs); test_cost  = np.zeros(nepochs)
		for i in xrange(nepochs):
			# Compute training cost
			self.timers.start_timers('train', 'train_epoch')
			self.enable_learning()
			train_cost[i] = cost(_run(train_data, train_labels), train_labels)
			self.timers.pause_timers('train')
			self.timers.stop_timers('train_epoch')
			if verbose: self._print_train(i, nepochs, train_cost)
			
			# Compute testing cost
			self.timers.start_timers('test', 'test_epoch')
			self.disable_learning()
			test_cost[i] = cost(_run(test_data), test_labels)
			self.timers.pause_timers('test')
			self.timers.stop_timers('test_epoch')
			if verbose: self._print_test(i, test_cost)
		
		self.timers.stop_timers('global')
		if verbose: self._print_final(nepochs, train_cost, test_cost)
		
		return (train_cost, test_cost)

class UnsupervisedCostNet(SupervisedCostNet):
	"""
	Base class description for network that computes a cost function using
	unsupervised training.
	"""
	__metaclass__ = ABCMeta
	
	def run(self, train_data, test_data, nepochs=1, verbose=True):
		"""
		Simulate the entire network.
		
		@param train_data: The data to train with. This must be an iterable
		returning a numpy array.
		
		@param test_data: The data to test with. This must be an iterable
		returning a numpy array.
		
		@param nepochs: The number of training epochs to perform.
		
		@param verbose: If True, details will be printed after each epoch.
		
		@return: A tuple containing the training and test costs.
		"""
		
		# Make some timers
		self.timers = MultiTimer()
		self.timers.add_timers('global', 'train', 'train_epoch', 'test',
			'test_epoch')
		self.timers.stop_timers('train', 'train_epoch', 'test', 'test_epoch')
		
		_run = self._run; cost = self.cost
		train_cost = np.zeros(nepochs); test_cost  = np.zeros(nepochs)
		for i in xrange(nepochs):
			# Compute training cost
			self.timers.start_timers('train', 'train_epoch')
			self.enable_learning()
			train_cost[i] = cost(_run(train_data), train_data)
			self.timers.pause_timers('train')
			self.timers.stop_timers('train_epoch')
			if verbose: self._print_train(i, nepochs, train_cost)
			
			# Compute testing cost
			self.timers.start_timers('test', 'test_epoch')
			self.disable_learning()
			test_cost[i] = cost(_run(test_data), test_data)
			self.timers.pause_timers('test')
			self.timers.stop_timers('test_epoch')
			if verbose: self._print_test(i, test_cost)
		
		self.timers.stop_timers('global')
		if verbose: self._print_final(nepochs, train_cost, test_cost)
		
		return (train_cost, test_cost)

class SupervisedAccuracyNet(Net):
	"""
	Base class description for network that computes an accuracy using
	supervised training.
	"""
	__metaclass__ = ABCMeta
	
	@abstractmethod
	def score(self, y, y_exp):
		"""
		Compute the accuracy. If there is competition between which node should
		be the winner, a random one is chosen.
		
		@param y: The true output.
		
		@param y_exp: The expected output.
		
		@return: The accuracy.
		"""
	
	def _print_train(self, epoch, nepochs, train_accuracy):
		"""
		Print the details about training.
		
		@param epoch: The current epoch number.
		
		@param nepochs: The total number of epochs.
		
		@param train_accuracy: A numpy array containing the training accuracy
		at each epoch.
		"""
		
		print '\nEpoch {0} of {1}:'.format(epoch + 1, nepochs)
		print '  Training Accuracy : {0}%'.format(train_accuracy[epoch] * 100)
		print '  Training Time     : {0}'.format(
			self.timers.get_elapsed_time('train_epoch', True))
	
	def _print_test(self, epoch, test_accuracy):
		"""
		Print the details about testing.
		
		@param epoch: The current epoch number.
		
		@param test_accuracy: A numpy array containing the testing accuracy
		at each epoch.
		"""
		
		print '  Testing Accuracy  : {0}%'.format(test_accuracy[epoch] * 100)
		print '  Testing Time      : {0}'.format(
			self.timers.get_elapsed_time('test_epoch', True))
	
	def _print_final(self, nepochs, train_accuracy, test_accuracy):
		"""
		Print the details final details.
		
		@param nepochs: The total number of epochs.
		
		@param train_accuracy: A numpy array containing the training accuracy
		at each epoch.
		
		@param test_accuracy: A numpy array containing the testing accuracy
		at each epoch.
		"""
		
		print '\n' + '*' * 79
		print '\nBest Training Accuracy : {0}% at Epoch {1}'.format(np.max(
			train_accuracy) * 100, np.argmax(train_accuracy) + 1)
		print 'Best Testing Accuracy  : {0}% at Epoch {1}'.format(np.max(
			test_accuracy) * 100, np.argmax(test_accuracy) + 1)
		print '\nTotal Execution Time        : {0}'.format(
			self.timers.get_elapsed_time('global', True))
		print 'Total Training Time         : {0}'.format(
			self.timers.get_elapsed_time('train', True))
		print 'Average Training Epoch Time : {0}'.format(
			pretty_time(self.timers.get_elapsed_time('train') / nepochs))
		print 'Total Testing Time          : {0}'.format(
			self.timers.get_elapsed_time('test', True))
		print 'Average Testing Epoch Time  : {0}'.format(
			pretty_time(self.timers.get_elapsed_time('test') / nepochs))
	
	def run(self, train_data, train_labels, test_data, test_labels,
		nepochs=1, verbose=True):
		"""
		Simulate the entire network.
		
		@param train_data: The data to train with. This must be an iterable
		returning a numpy array.
		
		@param train_labels: The training labels. This must be an iterable
		with the same length as train_data.
		
		@param test_data: The data to test with. This must be an iterable
		returning a numpy array.
		
		@param test_labels: The testing labels. This must be an iterable
		with the same length as train_data.
		
		@param nepochs: The number of training epochs to perform.
		
		@param verbose: If True, details will be printed after each epoch.
		
		@return: A tuple containing the training and test accuracies.
		"""
		
		# Make some timers
		self.timers = MultiTimer()
		self.timers.add_timers('global', 'train', 'train_epoch', 'test',
			'test_epoch')
		self.timers.stop_timers('train', 'train_epoch', 'test', 'test_epoch')
		
		_run = self._run; score = self.score
		train_accuracy = np.zeros(nepochs); test_accuracy  = np.zeros(nepochs)
		for i in xrange(nepochs):
			# Compute training accuracy
			self.timers.start_timers('train', 'train_epoch')
			self.enable_learning()
			train_accuracy[i] = score(_run(train_data, train_labels),
				train_labels)
			self.timers.pause_timers('train')
			self.timers.stop_timers('train_epoch')
			if verbose: self._print_train(i, nepochs, train_accuracy)
			
			# Compute testing accuracy
			self.timers.start_timers('test', 'test_epoch')
			self.disable_learning()
			test_accuracy[i] = score(_run(test_data), test_labels)
			self.timers.pause_timers('test')
			self.timers.stop_timers('test_epoch')
			if verbose: self._print_test(i, test_accuracy)
		
		self.timers.stop_timers('global')
		if verbose: self._print_final(nepochs, train_accuracy, test_accuracy)
		
		return (train_accuracy, test_accuracy)

###############################################################################
########## Class Implementation
###############################################################################

class LinearRegressionNetwork(SupervisedCostNet):
	"""
	Base class for a liner regression network.
	"""
	
	def __init__(self, ninputs, bias=1, learning_rate=0.001, min_weight=-1,
		max_weight=1, activation_type='linear', activation_kargs={},
		learning=True):
		"""
		Initializes this linear regression network.
		
		@param ninputs: The number of inputs (excluding the bias).
		
		@param bias: The bias input. Set to "0" to disable.
		
		@param learning_rate: The learning rate to use.
		
		@param min_weight: The minimum weight value.
		
		@param max_weight: The maximum weight value.
		
		@param activation_type: The type activation function to use. This must
		be one of the classes implemented in L{annt.activation}.
		
		@param activation_kargs: Any keyword arguments for the activation
		function.
		
		@param learning: Boolean denoting if the network is currently learning.
		"""
		
		# Store the params
		self.bias          = np.array([bias])
		self.learning_rate = learning_rate
		self.learning      = learning
		
		# Initialize the activation function
		self.activation = create_activation(activation_type,
			**activation_kargs)
		
		# Construct the weights
		self.initialize_weights((ninputs + 1,), min_weight, max_weight)
	
	def initialize_weights(self, shape, min_weight=-1, max_weight=1):
		"""
		Initialize the weights of the network. Initialization is done randomly.
		
		@param shape: The number of nodes in the entire network, excluding any
		bias terms. This parameter must be a sequence.
		
		@param min_weight: The minimum weight value.
		
		@param max_weight: The maximum weight value.
		"""
		
		self.weights = np.random.uniform(min_weight, max_weight, shape[0])
	
	def cost(self, y, y_exp):
		"""
		Compute the cost function
		
		@param y: The true output.
		
		@param y_exp: The expected output.
		
		@return: The cost.
		"""
		
		return np.sum(np.power((y_exp - y), 2)) / (2. * y_exp.shape[0])
	
	def step(self, x, y=None):
		"""
		Compute a single step of the network.
		
		@param x: The input data to compute for this step. This must be a numpy
		array with a shape of (self.ninputs, ).
		
		@param y: The expected output.
		
		@return: The computed output.
		"""
		
		# Add bias
		full_x = np.concatenate((self.bias, x))
		
		# Calculate the outputs
		y_est = self.activation.compute(np.dot(full_x, self.weights))
		
		# Update the error using online learning
		if self.learning:
			self.weights += self.learning_rate * full_x * (y - y_est)
		
		return y_est

class MultilayerPerception(SupervisedAccuracyNet):
	"""
	Base class for a multilayer perception.
	"""
	
	def __init__(self, shape, bias=1, learning_rate=0.001, min_weight=-1,
		max_weight=1, input_activation_type='linear',
		input_activation_kargs={'m':1}, hidden_activation_type='sigmoid',
		hidden_activation_kargs={},
		learning=True):
		"""
		Initializes this multilayer perception network.
		
		@param shape: The number of layers and the number of nodes per layer
		(excluding the bias).
		
		@param bias: The bias input. This is applied to all input and hidden
		layers. Set to "0" to disable.
		
		@param learning_rate: The learning rate to use.
		
		@param min_weight: The minimum weight value.
		
		@param max_weight: The maximum weight value.
		
		@param input_activation_type: The type activation function to use for
		the input layer. This must be one of the classes implemented in
		L{annt.activation}.
		
		@param input_activation_kargs: Any keyword arguments for the input
		activation function.
		
		@param hidden_activation_type: The type activation function to use for
		the hidden layer. This must be one of the classes implemented in
		L{annt.activation}.
		
		@param hidden_activation_kargs: Any keyword arguments for the hidden
		activation function.
		
		@param learning: Boolean denoting if the network is currently learning.
		"""
		
		# Store the params
		self.bias          = np.array([bias])
		self.learning_rate = learning_rate
		self.learning      = learning
		
		# Initialize the activation functions
		self.input_activation = create_activation(input_activation_type,
			**input_activation_kargs)
		self.hidden_activation = create_activation(hidden_activation_type,
			**hidden_activation_kargs)
		
		# Construct the weights
		self.initialize_weights(shape, min_weight, max_weight)
		
		# Construct the internal outputs and deltas
		new_shape    = [1 + s for s in shape[:-1]] + [shape[-1]]
		self.outputs = np.array([np.zeros(s) for s in new_shape])
		self.deltas  = self.outputs.copy()
	
	def initialize_weights(self, shape, min_weight=-1, max_weight=1):
		"""
		Initialize the weights of the network. Initialization is done randomly.
		
		@param shape: The number of nodes in the entire network, excluding any
		bias terms. This parameter must be a sequence.
		
		@param min_weight: The minimum weight value.
		
		@param max_weight: The maximum weight value.
		"""
		
		# Input weights aren't trained, so they are ignored. All other weights
		# are set to be random. The Last dimension is incremented by 1 to allow
		# for the bias.
		self.weights = np.array([np.random.uniform(min_weight, max_weight,
			(c, p + 1)) for c, p in zip(shape[1:], shape[:-1])])
	
	def score(self, y, y_exp):
		"""
		Compute the accuracy. If there is competition between which node should
		be the winner, a random one is chosen.
		
		@param y: The true output.
		
		@param y_exp: The expected output.
		
		@return: The accuracy.
		"""
		
		accuracy = 0.
		for predicted, expected in zip(y, y_exp):
			indexes = np.where(predicted == np.max(predicted))[0]
			np.random.shuffle(indexes)
			accuracy += 1 if expected[indexes[0]] == 1 else 0
		return accuracy / y_exp.shape[0]
	
	def step(self, x, y=None):
		"""
		Compute a single step of the network.
		
		@param x: The input data to compute for this step.
		
		@param y: The expected output.
		"""
		
		#######################################################################
		######## Calculate the outputs using forward propagation
		#######################################################################
		
		# Calculate the outputs for the input layer
		self.outputs[0][0]  = self.input_activation.compute(self.bias)
		self.outputs[0][1:] = self.input_activation.compute(x)
		
		# Calculate the outputs for the hidden layer(s)
		#   - First hidden layer -> last hidden layer
		for layer, layer_weights in enumerate(self.weights[:-1], 1):
			self.outputs[layer][0]  = self.hidden_activation.compute(self.bias)
			self.outputs[layer][1:] = self.hidden_activation.compute(np.inner(
				self.outputs[layer - 1], layer_weights))
		
		# Calculate the outputs for the output layer
		self.outputs[-1] = self.hidden_activation.compute(np.inner(
				self.outputs[-2], self.weights[-1]))
		
		#######################################################################
		######## Train the network using backpropagation
		#######################################################################
		
		if self.learning:
			
			###################################################################
			######## Calculate the deltas
			###################################################################
		
			# Calculate output deltas
			self.deltas[-1] = self.outputs[-1] - y
			
			# Calculate hidden deltas
			#   - Last hidden layer -> first hidden layer
			# Note that deltas are not computed for the bias
			for layer in xrange(-2, -self.deltas.shape[0], -1):
				self.deltas[layer] = self.hidden_activation.compute_derivative(
					self.outputs[layer][1:,]) * np.inner(self.deltas[layer +
					1], self.weights[layer + 1].T[1:,])
			
			###################################################################
			######## Update the weights
			###################################################################
			
			# Update the weights
			#   - Output -> first hidden layer
			#     - Bias's weight -> last node's weight
			for layer in xrange(-1, -self.deltas.shape[0], -1):
				for i in xrange(self.weights[layer].shape[0]):
					self.weights[layer][i] += -self.learning_rate *           \
						self.deltas[layer][i] * self.outputs[layer - 1]
		
		# Return the outputs from the output layer
		return self.outputs[-1]

class ExtremeLearningMachine(MultilayerPerception):
	"""
	Base class for an extreme learning machine.
	"""
		
	def step(self, x, y=None):
		"""
		Compute a single step of the network.
		
		@param x: The input data to compute for this step.
		
		@param y: The expected output.
		"""
		
		#######################################################################
		######## Calculate the outputs using forward propagation
		#######################################################################
		
		# Calculate the outputs for the input layer
		self.outputs[0][0]  = self.input_activation.compute(self.bias)
		self.outputs[0][1:] = self.input_activation.compute(x)
		
		# Calculate the outputs for the hidden layer(s)
		#   - First hidden layer -> last hidden layer
		for layer, layer_weights in enumerate(self.weights[:-1], 1):
			self.outputs[layer][0]  = self.hidden_activation.compute(self.bias)
			self.outputs[layer][1:] = self.hidden_activation.compute(np.inner(
				self.outputs[layer - 1], layer_weights))
		
		# Calculate the outputs for the output layer
		self.outputs[-1] = self.hidden_activation.compute(np.inner(
				self.outputs[-2], self.weights[-1]))
		
		#######################################################################
		######## Train the network using backpropagation
		#######################################################################
		
		if self.learning:
			
			###################################################################
			######## Calculate the deltas
			###################################################################
		
			# Calculate output deltas
			self.deltas[-1] = self.outputs[-1] - y
			
			###################################################################
			######## Update the weights
			###################################################################
			
			# Update the output weights
			for i in xrange(self.weights[-1].shape[0]):
				self.weights[-1][i] += -self.learning_rate *                  \
					self.deltas[-1][i] * self.outputs[-2]
		
		# Return the outputs from the output layer
		return self.outputs[-1]

class CompetitiveLearning(UnsupervisedCostNet):
	"""
	Base class for a competitive learning network (clustering).
	"""
	
	def __init__(self, ninputs, nclusters, learning_rate=0.001, boost_inc=0.1,
		boost_dec=0.01, duty_cycle=50, min_duty_cycle=5, min_weight=-1,
		max_weight=1, learning=True):
		"""
		Initializes this competitive learning network.
		
		@param ninputs: The number of inputs to the network.
		
		@param nclusters: The number of clusters.
		
		@param learning_rate: The learning rate to use.
		
		@param boost_inc: The amount to increment the boost by.
		
		@param boost_dec: The amount to decrement the boost by.
		
		@param duty_cycle: The history to retain for activations for each node.
		This is the period minimum activation is compared across. It is a
		rolling window.
		
		@param min_duty_cycle: The minimum duty cycle. If a node has not been
		active at least this many times, increment its boost value, else
		decrement it.
		
		@param min_weight: The minimum weight value.
		
		@param max_weight: The maximum weight value.
		
		@param learning: Boolean denoting if the network is currently learning.
		"""
		
		# Store the params
		self.learning_rate  = learning_rate
		self.boost_inc      = boost_inc
		self.boost_dec      = boost_dec
		self.duty_cycle     = duty_cycle
		self.min_duty_cycle = min_duty_cycle
		self.learning       = learning
		
		# Construct the weights
		self.initialize_weights((ninputs, nclusters), min_weight, max_weight)
		
		# Construct the boost values
		self.boost = np.ones(nclusters)
		
		# Construct the outputs
		#   - Each item represents a single cluster.
		#   - Each cluster maintains a history of length two of its update.
		#      - The first item refers to the current iteration.
		#      - The second item refers to the previous iteration.
		#      - The last item refers to the furthest iteration.
		self.outputs = np.ones((nclusters, duty_cycle))
	
	def _update_boost(self):
		"""
		Update the boost values.
		"""
		
		for i, active in enumerate(self.outputs):
			if int(np.sum(active)) >= self.min_duty_cycle:
				self.boost[i] += self.boost_inc
			else:
				self.boost[i] = max(self.boost[i] - self.boost_dec, 0)
	
	def initialize_weights(self, shape, min_weight=-1, max_weight=1):
		"""
		Initialize the weights of the network. Initialization is done randomly.
		
		@param shape: The number of nodes in the entire network. This parameter
		must be a sequence.
		
		@param min_weight: The minimum weight value.
		
		@param max_weight: The maximum weight value.
		"""
		
		self.weights = np.random.uniform(min_weight, max_weight, shape)
	
	def cost(self, y, x):
		"""
		Compute the cost function
		
		@param y: The true output.
		
		@param x: The input.
		
		@return: The cost.
		"""
		
		cost = 0.
		for i, pattern in enumerate(x):
			for j, weights in enumerate(self.weights.T):
				cost += y[i][j] * np.sum((weights - pattern) ** 2)
		return cost / 2
	
	def step(self, x):
		"""
		Compute a single step of the network.
		
		@param x: The input data to compute for this step.
		"""
		
		# Shit outputs
		self.outputs = np.roll(self.outputs, 1, 1)
		
		# Calculate the outputs
		for i, weights in enumerate(self.weights.T):
			self.outputs[i][0] = self.boost[i] * np.sum((weights - x) ** 2)
		min_ix                  = np.argmin(self.outputs.T[0])
		self.outputs.T[0]       = 0
		self.outputs[min_ix][0] = 1
		
		# Update the boosts
		self._update_boost()
		
		# Train the network
		if self.learning:
			for i, weights in enumerate(self.weights.T):
				self.weights.T[i] += self.learning_rate * self.outputs[i][0]  \
					* (x - weights)
		
		return self.outputs.T[0]