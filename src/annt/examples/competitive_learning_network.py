# competitive_learning_network.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Date Created   : 04/02/15
#	
# Description    : Example showing how to create and use a competitive learning
# network.
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
Example showing how to create and use a competitive learning network. This
example uses a reduced set of data from the
U{MNIST<http://yann.lecun.com/exdb/mnist/>} dataset.

G{packagetree annt}
"""

__docformat__ = 'epytext'

# Program imports
from annt.util import one_hot, mnist_data
from annt.net  import CompetitiveLearning
from annt.plot import basic_epoch

def main(train_data, train_labels, test_data, test_labels, nepochs=1):
	"""
	Demonstrates a competitive learning network using MNIST.
	
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
	net =  CompetitiveLearning(
		ninputs       = 784,
		nclusters     = 10,
		learning_rate = 0.001,
		min_weight    = -1,
		max_weight    = 1
	)
	
	# Simulate the network
	train_cost, test_cost = net.run(train_data, train_labels, test_data,
		test_labels, nepochs, True)
	
	# Plot the results
	basic_epoch((train_cost, test_cost), ('Train', 'Test'), 'Cost',
		'Clustering - Example', semilog=True)

if __name__ == '__main__':
	# Get the data
	(train_data, train_labels), (test_data, test_labels) = mnist_data()
	
	# Scale pixel values to be between 0 and 1
	# Convert labeled data to one-hot encoding
	# Run the network
	main(train_data/255., one_hot(train_labels, 10), test_data/255.,
		one_hot(test_labels, 10), nepochs=100)