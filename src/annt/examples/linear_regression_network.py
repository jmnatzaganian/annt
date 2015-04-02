# linear_regression_network.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Date Created   : 03/31/15
#	
# Description    : Example showing how to create and use a linear regression
# network.
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
Example showing how to create and use a linear regression network. This example
uses a reduced set of data from the U{MNIST<http://yann.lecun.com/exdb/mnist/>}
dataset.

G{packagetree annt}
"""

__docformat__ = 'epytext'

# Third party imports
import numpy as np

# Program imports
from annt.util import mnist_data
from annt.net  import LinearRegressionNetwork
from annt.plot import basic_epoch

def main(train_data, train_labels, test_data, test_labels, nepochs=1):
	"""
	Demonstrates a linear regression network using MNIST.
	
	@param train_data: The data to train with. This must be an iterable
	returning a numpy array.
	
	@param train_labels: The training labels. This must be an iterable with the
	same length as train_data.
	
	@param test_data: The data to test with. This must be an iterable returning
	a numpy array.
	
	@param test_labels: The testing labels. This must be an iterable with the
	same length as train_data.
	
	@param nepochs: The number of training epochs to perform.
	"""
	
	# Create the network
	net = LinearRegressionNetwork(
		ninputs          = train_data.shape[1],
		bias             = 1,
		learning_rate    = 0.001,
		min_weight       = -1,
		max_weight       = 1,
		activation_type  = 'linear',
		activation_kargs = {'m':1}
	)
	
	# Simulate the network
	train_cost, test_cost = net.run(train_data, train_labels, test_data,
		test_labels, nepochs)
	
	print '\nBest Training Cost:\n{0} at Epoch {1}'.format(np.min(train_cost),
		np.argmin(train_cost))
	print '\nBest Testing Cost:\n{0} at Epoch {1}'.format(np.min(test_cost),
		np.argmin(test_cost))
	
	# Plot the results
	basic_epoch((train_cost, test_cost), ('Train', 'Test'), 'Cost',
		'Linear Regression Network - Example', semilog=True)

if __name__ == '__main__':
	# Get the data
	(train_data, train_labels), (test_data, test_labels) = mnist_data()
	
	# Scale pixel values to be between 0 and 1
	# Scale label values to be between 0 and 1
	# Run the network
	main(train_data/255., train_labels/9., test_data/255., test_labels/9.,
		nepochs=100)