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
from annt.util import mnist_data
from annt.net  import CompetitiveLearning
from annt.plot import basic_epoch, plot_weights

def main(train_data, test_data, nepochs=1):
	"""
	Demonstrates a competitive learning network using MNIST.
	
	@param train_data: The data to train with. This must be an iterable
	returning a numpy array.
	
	@param test_data: The data to test with. This must be an iterable returning
	a numpy array.
	
	@param nepochs: The number of training epochs to perform.
	"""
	
	# Create the network
	net =  CompetitiveLearning(
		ninputs        = 784,
		nclusters      = 10,
		learning_rate  = 0.001,
		boost_inc      = 0.1,
		boost_dec      = 0.01,
		duty_cycle     = 50,
		min_duty_cycle = 5,
		min_weight     = -1,
		max_weight     = 1
	)
	
	# Simulate the network
	train_cost, test_cost = net.run(train_data, test_data, nepochs, True)
	
	# Plot the results
	basic_epoch((train_cost, test_cost), ('Train', 'Test'), 'Cost',
		'Clustering - Example', semilog=True)
	
	# Plot clusters
	plot_weights(net.weights.T, 2, 5, (28, 28), 'Clustering Weights - Example')

if __name__ == '__main__':
	# Get the data
	(train_data, train_labels), (test_data, test_labels) = mnist_data()
	
	# Scale pixel values to be between 0 and 1
	# Run the network
	main(train_data/255., test_data/255., nepochs=50)