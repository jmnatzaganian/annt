# hopfield_net.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Date Created   : 04/07/15
#	
# Description    : Module for a Hopfield network.
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
Module for a Hopfield network.

G{packagetree annt}
"""

__docformat__ = 'epytext'

# Third party imports
import numpy as np

# Program imports
from annt.activation import create_activation
from annt.util       import get_random_paired_indexes, threshold

###############################################################################
########## Class Implementation
###############################################################################

class Hopfield(object):
	"""
	Base class for a Hopfield network.
	"""
	
	def __init__(self, activation_type='sigmoid', activation_kargs={},
		attractor_states=None):
		"""
		Initializes this Hopfield network.
			
		@param activation_type: The type activation function to use for the
		neurons. This must be one of the classes implemented in
		L{annt.activation}.
		
		@param activation_kargs: Any keyword arguments for the activation
		function.
		
		@param attractor_states: The attractor states to use. This must be a
		dictionary containing a unique state as the key (this should be a tuple
		containing a vector of representing the current state). The value in
		the dictionary should be the corresponding label of the state. If None,
		the attractor states must be created using the
		"create_attractor_states" method.
		"""
		
		# Store the params
		self.attractor_states = attractor_states
		
		# Initialize the activation function
		self.activation = create_activation(activation_type,
			**activation_kargs)
		
		# Construct the weights (if possible)
		if self.attractor_states is not None:
			self.initialize_weights()
	
	def create_attractor_states(self, x, y, nstates, nsamples, thresh):
		"""
		Create the attractor states based off the labeled data. Additionally,
		initialize the weights using the new attractor states.
		
		@param x: A numpy array consisting of the data to initialize with.
		
		@param y: A numpy array consisting of labels.
		
		@param nstates: The number of states to create. This will result in
		creating a number of attractor states equal to the amount of unique
		labels multiplied by this value.
		
		@param nsamples: The number of samples to use for each unique value in
		y.
		
		@param thresh: The value to threshold at.
		"""
		
		idxs                  = [get_random_paired_indexes(y, nsamples)
			for _ in xrange(nstates)]
		self.attractor_states = {}
		for idx in idxs:
			for i, lbl in enumerate(idx):
				states = [x[ix] for ix in idx[lbl]]
				self.attractor_states[tuple(threshold(np.mean(states, 0), 0))
					] = lbl
		
		self.initialize_weights()
	
	def initialize_weights(self):
		"""
		Initialize the weights of the network. Initialization is done based off
		the attractor states.
		"""
		
		states       = np.array(self.attractor_states.keys()).T
		self.weights = np.zeros((states.shape[0], states.shape[0]))		
		for i in xrange(states.shape[0]):
			for j in xrange(states.shape[0]):
					self.weights[i][j] = np.dot(states[i], states[j]) / float(
						states.shape[0])
	
	def step(self, x):
		"""
		Compute a single step of the network.
		
		@param x: The input data for this step.
		
		@return: The found attractor state.
		"""
		
		return threshold(np.inner(self.weights, x), 0)