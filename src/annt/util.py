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

###############################################################################
########## Primary Functions
###############################################################################

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