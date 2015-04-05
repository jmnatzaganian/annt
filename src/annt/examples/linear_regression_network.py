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

# Native imports
import os

# Third party imports
import numpy as np

# Program imports
from annt.util import mnist_data
from annt.net  import LinearRegressionNetwork
from annt.plot import plot_epoch

def main(train_data, train_labels, test_data, test_labels, nepochs=1,
	plot=True, verbose=True, bias=1, learning_rate=0.001, min_weight=-1,
	max_weight=1, activation_type='linear', activation_kargs={'m':1}):
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
	
	@param plot: If True, a plot will be created.
	
	@param verbose: If True, the network will print results after every
	iteration.
	
	@param bias: The bias input. Set to "0" to disable.
		
	@param learning_rate: The learning rate to use.
	
	@param min_weight: The minimum weight value.
	
	@param max_weight: The maximum weight value.
	
	@param activation_type: The type activation function to use. This must be
	one of the classes implemented in L{annt.activation}.
	
	@param activation_kargs: Any keyword arguments for the activation
	function.
	"""
	
	# Create the network
	net = LinearRegressionNetwork(
		ninputs          = train_data.shape[1],
		bias             = bias,
		learning_rate    = learning_rate,
		min_weight       = min_weight,
		max_weight       = max_weight,
		activation_type  = activation_type,
		activation_kargs = activation_kargs
	)
	
	# Simulate the network
	train_results, test_results = net.run(train_data, train_labels, test_data,
		test_labels, nepochs, verbose)	
	
	# Plot the results
	if plot:
		plot_epoch(y_series=(train_results, test_results),
			series_names=('Train', 'Test'), y_label='Cost',
			title='Linear Regression Network - Example', semilog=True)
	
	return train_results, test_results

def basic_sim(nepochs=100):
	"""
	Perform a basic simulation.
	
	@param nepochs: The number of training epochs to perform.
	"""
	
	# Get the data
	(train_data, train_labels), (test_data, test_labels) = mnist_data()
	
	# Scale pixel values to be between 0 and 1
	# Scale label values to be between 0 and 1
	# Run the network
	main(train_data/255., train_labels/9., test_data/255., test_labels/9.,
		nepochs=100)

def bulk(niters, nepochs, verbose=True, plot=True, **kargs):
	"""
	Execute the main network across many networks.
	
	@param niters: The number of iterations to run for statistical purposes.
	
	@param nepochs: The number of training epochs to perform.
	
	@param verbose: If True, a simple iteration status will be printed.
	
	@param plot: If True, a plot will be generated.
	
	@param kargs: Any keyword arguments to pass to the main network simulation.
	
	@return: A tuple containing: (train_mean, train_std), (test_mean, test_std)
	"""
	
	# Simulate the network
	train_results = np.zeros((niters, nepochs))
	test_results  = np.zeros((niters, nepochs))
	for i in xrange(niters):
		if verbose:
			print 'Executing iteration {0} of {1}'.format(i + 1, niters)
		train_results[i], test_results[i] = main(verbose=False, plot=False,
			nepochs=nepochs, **kargs)
	
	# Compute the mean costs
	train_mean = np.mean(train_results, 0)
	test_mean  = np.mean(test_results, 0)
	
	# Compute the standard deviations
	train_std = np.std(train_results, 0)
	test_std  = np.std(test_results, 0)
	
	if plot:
		plot_epoch(y_series=(train_mean, test_mean), semilog=True,
			series_names=('Train', 'Test'), y_errs=(train_std, test_std),
			y_label='Cost',	title='Linear Regression Network - Stats Example')
	
	return (train_mean, train_std), (test_mean, test_std)

def bulk_sim(nepochs=100, niters=10):
	"""
	Perform a simulation across multiple iterations, for statistical purposes.
	
	@param nepochs: The number of training epochs to perform.
	
	@param niters: The number of iterations to run for statistical purposes.
	"""
	
	(train_data, train_labels), (test_data, test_labels) = mnist_data()
	bulk(train_data=train_data/255., train_labels=train_labels/9.,
		test_data=test_data/255., test_labels=test_labels/9., nepochs=nepochs,
		niters=niters)

def vary_params(out_dir, nepochs=100, niters=10):
	"""
	Vary some parameters and generate some plots.
	
	@param out_dir: The directory to save the plots in.
	
	@param nepochs: The number of training epochs to perform.
	
	@param niters: The number of iterations to run for statistical purposes.
	"""
	
	# Get the data
	(train_data, train_labels), (test_data, test_labels) = mnist_data()
	
	# Make the output directory
	try:
		os.makedirs(out_dir)
	except OSError:
		pass
	
	###########################################################################
	###### Vary learning rate
	###########################################################################
	
	print 'Varying the learning rate'
	learning_rates = np.linspace(0.001, 0.01, 10)
	train_results  = np.zeros((learning_rates.shape[0], nepochs))
	train_stds     = np.zeros((learning_rates.shape[0], nepochs))
	test_results   = np.zeros((learning_rates.shape[0], nepochs))
	test_stds      = np.zeros((learning_rates.shape[0], nepochs))
	series_names   = ['Learning Rate = {0}'.format(x) for x in learning_rates]
	for i, learning_rate in enumerate(learning_rates):
		print 'Executing iteration {0} of {1}'.format(i + 1,
			learning_rates.shape[0])
		(train_results[i], train_stds[i]), (test_results[i], test_stds[i]) =  \
			bulk(train_data=train_data/255., train_labels=train_labels/9.,
			test_data=test_data/255., test_labels=test_labels/9., plot=False,
			nepochs=nepochs, niters=niters, learning_rate=learning_rate,
			verbose=False)
	
	# Make training plot
	title    = 'Linear Regression Network - Training\n10 Iterations, '        \
		'Varying Learning Rate'
	out_path = os.path.join(out_dir, 'learning_rate-train.png')
	plot_epoch(y_series=train_results, semilog=True, series_names=series_names,
		y_errs=train_stds, y_label='Cost', title=title, out_path=out_path)
	
	# Make testing plot
	title    = 'Linear Regression Network - Testing\n10 Iterations, '         \
		'Varying Learning Rate'
	out_path = os.path.join(out_dir, 'learning_rate-test.png')
	plot_epoch(y_series=test_results, semilog=True, series_names=series_names,
		y_errs=train_stds, y_label='Cost', title=title, out_path=out_path)
	
	###########################################################################
	###### Vary slope
	###########################################################################
	
	print '\nVarying the slope of the linear function'
	slopes         = np.linspace(1, 10, 10)
	train_results  = np.zeros((learning_rates.shape[0], nepochs))
	train_stds     = np.zeros((learning_rates.shape[0], nepochs))
	test_results   = np.zeros((learning_rates.shape[0], nepochs))
	test_stds      = np.zeros((learning_rates.shape[0], nepochs))
	series_names   = ['Slope = {0}'.format(x) for x in slopes]
	for i, slope in enumerate(slopes):
		print 'Executing iteration {0} of {1}'.format(i + 1,
			slopes.shape[0])
		(train_results[i], train_stds[i]), (test_results[i], test_stds[i]) =  \
			bulk(train_data=train_data/255., train_labels=train_labels/9.,
			test_data=test_data/255., test_labels=test_labels/9., plot=False,
			nepochs=nepochs, niters=niters, activation_kargs={'m':slope},
			verbose=False)
	
	# Make training plot
	title    = 'Linear Regression Network - Training\n10 Iterations, '        \
		"Varying Activation Function's Slope"
	out_path = os.path.join(out_dir, 'slope-train.png')
	plot_epoch(y_series=train_results, semilog=True, series_names=series_names,
		y_errs=train_stds, y_label='Cost', title=title, out_path=out_path)
	
	# Make testing plot
	title    = 'Linear Regression Network - Testing\n10 Iterations, '         \
		"Varying Activation Function's Slope"
	out_path = os.path.join(out_dir, 'slope-test.png')
	plot_epoch(y_series=test_results, semilog=True, series_names=series_names,
		y_errs=train_stds, y_label='Cost', title=title, out_path=out_path)

if __name__ == '__main__':
	# basic_sim()
	# bulk_sim()
	vary_params(out_dir=r'D:\annt\test')