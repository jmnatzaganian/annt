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