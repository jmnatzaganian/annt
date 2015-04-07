# hopfield_network.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Date Created   : 04/07/15
#	
# Description    : Example showing how to create and use a hopfield network.
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
Example showing how to create and use a hopfield network. This example uses a
reduced set of data from the U{MNIST<http://yann.lecun.com/exdb/mnist/>}
dataset.

G{packagetree annt}
"""

__docformat__ = 'epytext'

# Native imports
import os

# Third party imports
import numpy as np

# Program imports
from annt.util                      import mnist_data, threshold
from annt.experimental.hopfield_net import Hopfield
from annt.plot                      import plot_weights

def plot_attractor_states(net, nrows, ncols, shape, **kargs):
	"""
	Plot the attractor states.
	
	@param net: A Hopfield network instance.
	
	@param nrows: The number of rows of plots to create.
	
	@param ncols: The number of columns of plots to create.
	
	@param shape: The shape of the weights. It is assumed that a 1D shape was
	used and is desired to be represented in 2D. Whatever shape is provided
	will be used to reshape the weights. For example, if you had a 28x28 image
	and each weight corresponded to one pixel, you would have a vector with a
	shape of (784, ). This vector would then need to be resized to your desired
	shape of (28, 28).
	
	@param kargs: Any additional keyword arguments for the "plot_weights"
	function.
	"""
	
	plot_weights(np.array(net.attractor_states.keys()), nrows, ncols, shape,
		**kargs)

def main(train_data, train_labels, test_data, test_labels, plot=True,
	verbose=True, activation_type='sigmoid', activation_kargs={}, nsamples=10,
	nstates=1):
	"""
	Demonstrates a hopfield network using MNIST.
	
	@param train_data: The data to train with. This must be an iterable
	returning a numpy array.
	
	@param train_labels: The training labels. This must be an iterable with the
	same length as train_data.
	
	@param test_data: The data to test with. This must be an iterable returning
	a numpy array.
	
	@param test_labels: The testing labels. This must be an iterable with the
	same length as train_data.
	
	@param plot: If True, a plot will be created.
	
	@param verbose: If True, the network will print results after every
	iteration.
	
	@param activation_type: The type activation function to use for the
	neurons. This must be one of the classes implemented in
	L{annt.activation}.
	
	@param activation_kargs: Any keyword arguments for the activation function.
	
	@param nsamples: The number of samples to use for generating the attractor
	states.
	
	@param nstates: The number of states to create. This will result in
	creating a number of attractor states equal to the amount of unique labels
	multiplied by this value.
	
	@return: A tuple containing the training and testing results, respectively.
	"""
	
	# Create the network
	net   =  Hopfield(
		activation_type  = activation_type,
		activation_kargs = activation_kargs
	)
	
	# Initialize the attractor states
	net.create_attractor_states(train_data, train_labels, nstates, nsamples,
		255 / 2)
	
	# Simulate the network
	train_results, test_results = net.run(train_data, train_labels,
		test_data, test_labels, nepochs, verbose)
	
	# # Plot the results
	# if plot:
		# plot_epoch(y_series=(train_results * 100, test_results * 100),
			# series_names=('Train', 'Test'), y_label='Accuracy [%]',
			# title='MLP - Example', legend_location='upper left')
	
	# return train_results * 100, test_results * 100

def basic_sim():
	"""
	Perform a basic simulation.
	"""
	
	# Get the data
	(train_data, train_labels), (test_data, test_labels) = mnist_data()
	
	# Scale pixel values to be between -1 and 1
	# Run the network
	main(threshold(train_data, 255 / 2), train_labels,
		threshold(test_data, 255 / 2), test_labels, nepochs=nepochs)

def vary_params(out_dir, show_plot=True):
	"""
	Vary some parameters and generate some plots.
	
	@param out_dir: The directory to save the plots in.
	
	@param show_plot: If True the plot will be displayed upon creation.
	"""
	
	# Get the data
	(train_data, train_labels), (test_data, test_labels) = mnist_data()
	train_d = threshold(train_data, 255 / 2)
	test_d  = threshold(test_data, 255 / 2)
	
	# Make the output directory
	try:
		os.makedirs(out_dir)
	except OSError:
		pass
	
	###########################################################################
	###### Vary number of attractors
	###########################################################################
	
	
	
	###########################################################################
	###### Vary amount of noise on input
	###########################################################################
	
	

if __name__ == '__main__':
	basic_sim()
	# vary_params(out_dir=r'D:\annt\test\MLP_Network', show_plot=False)