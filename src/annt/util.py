# util.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Date Created   : 04/02/15
#	
# Description    : Utility module.
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
Utility module. This module handles any sort of accessory items.

G{packagetree annt}
"""

__docformat__ = 'epytext'

# Native imports
import os, pkgutil, cPickle

# Third party imports
import numpy as np

# Program imports
from annt.exception_handler import BaseException, wrap_error

###############################################################################
########## Exception Handling
###############################################################################

class InsufficientSamples(BaseException):
	"""
	Exception if too many samples were desired.
	"""
	
	def __init__(self, sample, navail, nsamples):
		"""
		Initialize this class.
		
		@param sample: The desired sample.
		
		@param navail: The number of available samples.
		
		@param nsamples: The desired number of samples to extract.
		"""
		
		self.msg = wrap_error('The sample, {0}, only has {1} instances. The ' \
			'requested number of samples of {2} is too large. Please reduce ' \
			'the desired number of samples and try again'.format(sample,
			navail, nsamples))

###############################################################################
########## Primary Functions
###############################################################################

def mnist_data():
	"""
	Return the example U{MNIST<http://yann.lecun.com/exdb/mnist/>} data. This
	is merely a subset of the data. There are 80 samples per digit for the
	training set (800 total items) and 20 samples per digit for the testing set
	(200 total items).
	
	@return: A tuple of tuples of the following format:
	(train_data, train_labels), (test_data, test_labels)
	"""
	
	with open(os.path.join(pkgutil.get_loader('annt.examples').filename,
		'data', 'mnist.pkl'), 'rb') as f:
		return cPickle.load(f)

def one_hot(x, num_items):
	"""
	Convert an array into a one-hot encoding. The indices in x mark which bits
	should be set. The length of each sub-array will be determined by
	num_items.
	
	@param x: The array indexes to mark as valid. This should be a numpy array.
	
	@param num_items: The number of items each encoding should contain. This
	should be at least as large as the max value in x + 1.
	
	@return: An encoded array.
	"""
	
	y = np.repeat([np.zeros(num_items, dtype='uint8')], x.shape[0], 0)
	for i, ix in enumerate(x):
		y[i, ix] = 1
	return y

def threshold(x, thresh, min_value=-1, max_value=1):
	"""
	Threshold all of the data in a given matrix (2D).
	
	@param x: The array to threshold.
	
	@param thresh: The value to threshold at.
	
	@param min_value: The minimum value to set the data to.
	
	@param max_value: The maximum value to set the data to.
	
	@return: An encoded array.
	"""
	
	y = np.empty(x.shape); y.fill(min_value)
	max_idx    = x >= thresh
	y[max_idx] = max_value
	return y

def get_random_paired_indexes(y, nsamples):
	"""
	Get a list of indexes corresponding to random selections of the data in y.
	A total of nsamples will be returned for each unique value in y.
	
	@param y: A numpy array consisting of labels.
	
	@param nsamples: The number of samples to obtain for each unique value in
	y.
	
	@return: A dictionary containing the indexes in y corresponding to each
	unique value in y. There will be a total of nsamples indexes.
	
	@raise InsufficientSamples: Raised if too many samples were desired to be
	selected.
	"""
	
	# Extract initial indexes
	keys = np.unique(y)
	idx  = [np.where(key == y)[0] for key in keys]
	
	# Check to make sure it is possible
	for key, ix in zip(keys, idx):
		if ix.shape[0] < nsamples:
			raise InsufficientSamples(key, ix.shape[0], nsamples)
	
	# Shuffle the indexes
	for i in xrange(len(idx)):
		np.random.shuffle(idx[i])
	
	# Build final result
	return {key:ix[:nsamples] for key, ix in zip(keys, idx)}