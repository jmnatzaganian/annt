# activation.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Date Created   : 03/30/15
#	
# Description    : Module for various activation functions.
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
Module for various activation functions.

G{packagetree annt}
"""

__docformat__ = 'epytext'

# Native imports
from abc import ABCMeta, abstractmethod

# Third party imports
import numpy as np

# Program imports
from annt.exception_handler import BaseException, wrap_error

###############################################################################
########## Exception Handling
###############################################################################

class UnsupportedActivationType(BaseException):
	"""
	Exception if the activation type is invalid.
	"""
	
	def __init__(self, type):
		"""
		Initialize this class.
		
		@param type: The type of activation function to use.
		"""
		
		self.msg = wrap_error('The type, {0}, is unsupported. The current '
			'types are {1}'.format(name, ', '.join(['linear', 'sigmoid'])))

###############################################################################
########## Functions
###############################################################################

def get_activation(type):
	"""
	Returns a reference to an activation object.
	
	@param type: The type of activation function to use.
	
	@return: An activation object reference.
	
	@raise UnsupportedActivationType: Raised if type is invalid.
	"""
	
	if type == 'linear':
		return Linear
	elif type == 'sigmoid':
		return Sigmoid
	else:
		raise UnsupportedActivationType(type)

def create_activation(type, **kargs):
	"""
	Creates an activation object instance.
	
	@param type: The type of activation function to use.
	
	@param kargs: Any keyword arguments.
	"""
	
	return get_activation(type)(**kargs)

###############################################################################
########## Class Template
###############################################################################

class Activation(object):
	"""
	Base class description for an activation function.
	"""
	__metaclass__ = ABCMeta
	
	@abstractmethod
	def __init__(self):
		"""
		Initialize the class instance.
		"""
	
	@abstractmethod
	def compute(self, x):
		"""
		Compute the activation function.
		
		@param x: A numpy array representing the input data. This should be a
		vector.
		
		@return: A vector containing the element-wise result of applying the
		activation function to the input.
		"""

###############################################################################
########## Class Implementation
###############################################################################

class Linear(Activation):
	"""
	Base class for a liner activation function.
	"""
	
	def __init__(self, m=1):
		"""
		Initializes this linear object.
		
		@param m: The slope of the line. Use the default value of "1" for the
		unity function.
		"""
		
		# Store the params
		self.m = m
	
	def compute(self, x):
		"""
		Compute the activation function.
		
		@param x: A numpy array representing the input data. This should be a
		vector.
		
		@return: A vector containing the element-wise result of applying the
		activation function to the input.
		"""
		
		return self.m * x
	
	def compute_derivative(self, x):
		"""
		Compute the activation function's derivative.
		
		@param x: A numpy array representing the input data. This should be a
		vector.
		
		@return: A vector containing the element-wise result of applying the
		activation function to the input.
		"""
		
		return np.repeat(self.m * len(x))

class Sigmoid(Activation):
	"""
	Base class for a sigmoid activation function.
	"""
	
	def __init__(self):
		"""
		Initializes this sigmoid object.
		"""
		
		pass
	
	def compute(self, x):
		"""
		Compute the activation function.
		
		@param x: A numpy array representing the input data. This should be a
		vector.
		
		@return: A vector containing the element-wise result of applying the
		activation function to the input.
		"""
		
		return 1 / (1 + np.exp(-x))
	
	def compute_derivative(self, x):
		"""
		Compute the activation function's derivative.
		
		@param x: A numpy array representing the input data. This should be a
		vector.
		
		@return: A vector containing the element-wise result of applying the
		activation function to the input.
		"""
		
		y = self.compute(x)
		
		return y * (1 - y)