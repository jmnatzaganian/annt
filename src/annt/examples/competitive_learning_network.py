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

# Native imports
import os

# Third party imports
import numpy as np

# Program imports
from annt.util import mnist_data
from annt.net  import CompetitiveLearning
from annt.plot import plot_epoch, plot_weights, plot_surface, make_grid

def main(train_data, test_data, nepochs=1, plot=True, verbose=True,
	nclusters=10, learning_rate=0.001, boost_inc=0.1, boost_dec=0.01,
	duty_cycle=50, min_duty_cycle=5, min_weight=-1, max_weight=1, nrows=2,
	ncols=5, shape=(28, 28)):
	"""
	Demonstrates a competitive learning network using MNIST.
	
	@param train_data: The data to train with. This must be an iterable
	returning a numpy array.
	
	@param test_data: The data to test with. This must be an iterable returning
	a numpy array.
	
	@param nepochs: The number of training epochs to perform.
	
	@param plot: If True, a plot will be created.
	
	@param verbose: If True, the network will print results after every
	iteration.
	
	@param nclusters: The number of clusters.
		
	@param learning_rate: The learning rate to use.
	
	@param boost_inc: The amount to increment the boost by.
	
	@param boost_dec: The amount to decrement the boost by.
	
	@param duty_cycle: The history to retain for activations for each node.
	This is the period minimum activation is compared across. It is a rolling 
	window.
	
	@param min_duty_cycle: The minimum duty cycle. If a node has not been 
	active at least this many times, increment its boost value, else decrement
	it.
	
	@param min_weight: The minimum weight value.
	
	@param max_weight: The maximum weight value.
	
	@param nrows: The number of rows of plots to create for the clusters.
	
	@param ncols: The number of columns of plots to create for the clusters.
	
	@param shape: The shape of the weights. It is assumed that a 1D shape was
	used and is desired to be represented in 2D. Whatever shape is provided
	will be used to reshape the weights. For example, if you had a 28x28 image
	and each weight corresponded to one pixel, you would have a vector with a
	shape of (784, ). This vector would then need to be resized to your desired
	shape of (28, 28).
	
	@return: A tuple containing the training results, testing results, and
	weights, respectively.
	"""
	
	# Create the network
	net =  CompetitiveLearning(
		ninputs        = train_data.shape[1],
		nclusters      = nclusters,
		learning_rate  = learning_rate,
		boost_inc      = boost_inc,
		boost_dec      = boost_dec,
		duty_cycle     = duty_cycle,
		min_duty_cycle = min_duty_cycle,
		min_weight     = min_weight,
		max_weight     = max_weight
	)
	
	# Simulate the network
	train_results, test_results = net.run(train_data, test_data, nepochs,
		verbose)
	
	# Plot the results and clusters
	if plot:
		plot_epoch(y_series=(train_results, test_results),
			series_names=('Train', 'Test'), y_label='Cost',
			title='Clustering - Example', semilog=True)
		plot_weights(net.weights.T, nrows, ncols, shape,
			'Clustering Weights - Example')
	
	return train_results, test_results, net.weights.T

def basic_sim(nepochs=100):
	"""
	Perform a basic simulation.
	
	@param nepochs: The number of training epochs to perform.
	"""
	
	# Get the data
	(train_data, train_labels), (test_data, test_labels) = mnist_data()
	
	# Scale pixel values to be between 0 and 1
	# Convert labeled data to one-hot encoding
	# Run the network
	main(train_data/255., test_data/255., nepochs=nepochs)

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
		train_results[i], test_results[i], _ = main(verbose=False,
			plot=False, nepochs=nepochs, **kargs)
	
	# Compute the mean costs
	train_mean   = np.mean(train_results, 0)
	test_mean    = np.mean(test_results, 0)
	
	# Compute the standard deviations
	train_std   = np.std(train_results, 0)
	test_std    = np.std(test_results, 0)
	
	if plot:
		plot_epoch(y_series=(train_mean, test_mean),
			legend_location='upper left', series_names=('Train', 'Test'),
			y_errs=(train_std, test_std), y_label='Cost [%]',
			title='Clustering - Stats Example')
	
	return (train_mean, train_std), (test_mean, test_std)

def bulk_sim(nepochs=100, niters=10):
	"""
	Perform a simulation across multiple iterations, for statistical purposes.
	
	@param nepochs: The number of training epochs to perform.
	
	@param niters: The number of iterations to run for statistical purposes.
	"""
	
	(train_data, train_labels), (test_data, test_labels) = mnist_data()
	bulk(train_data=train_data/255., test_data=test_data/255., nepochs=nepochs,
		niters=niters)

def vary_params(out_dir, nepochs=100, niters=10, show_plot=True):
	"""
	Vary some parameters and generate some plots.
	
	@param out_dir: The directory to save the plots in.
	
	@param nepochs: The number of training epochs to perform.
	
	@param niters: The number of iterations to run for statistical purposes.
	
	@param show_plot: If True the plot will be displayed upon creation.
	"""
	
	# Get the data
	(train_data, train_labels), (test_data, test_labels) = mnist_data()
	train_d = train_data/255.; test_d  = test_data/255.
	
	# Make the output directory
	try:
		os.makedirs(out_dir)
	except OSError:
		pass
	
	###########################################################################
	###### Vary number of clusters
	###########################################################################
	
	print 'Varying number of clusters'
	nclusters          = np.arange(5, 55, 5)
	weight_plot_params = [{'nrows':1, 'ncols':5}, {'nrows':2, 'ncols':5},
		{'nrows':3, 'ncols':5}, {'nrows':4, 'ncols':5}, {'nrows':5, 'ncols':5},
		{'nrows':3, 'ncols':10}, {'nrows':4, 'ncols':10},
		{'nrows':4, 'ncols':10}, {'nrows':5, 'ncols':10},
		{'nrows':5, 'ncols':10}]
	weight_results     = []
	train_results      = np.zeros((len(nclusters), nepochs))
	train_stds         = np.zeros((len(nclusters), nepochs))
	test_results       = np.zeros((len(nclusters), nepochs))
	test_stds          = np.zeros((len(nclusters), nepochs))
	series_names       = ['Clusters = {0}'.format(x) for x in nclusters]
	for i, ncluster in enumerate(nclusters):
		print 'Executing iteration {0} of {1}'.format(i + 1, len(nclusters))
		(train_results[i], train_stds[i]), (test_results[i], test_stds[i]) =  \
			bulk(train_data=train_d, test_data=test_d, plot=False,
			nepochs=nepochs, nclusters=ncluster, verbose=False, niters=niters)
		x, y, weights = main(train_data=train_d, test_data=test_d, plot=False,
			nepochs=nepochs, nclusters=ncluster, verbose=False)
		weight_results.append(weights)
	
	# Make training plot
	title    = 'Competitive Learning Network - Training\n10 Iterations, '     \
		'Varying Number of Clusters'
	out_path = os.path.join(out_dir, 'clusters-train.png')
	plot_epoch(y_series=train_results, series_names=series_names, title=title,
		y_errs=train_stds, y_label='Cost', out_path=out_path, semilog=True,
		show=show_plot)
	
	# Make testing plot
	title    = 'Competitive Learning Network - Testing\n10 Iterations, '      \
		'Varying Number of Clusters'
	out_path = os.path.join(out_dir, 'clusters-test.png')
	plot_epoch(y_series=test_results, series_names=series_names, title=title,
		y_errs=test_stds, y_label='Cost', out_path=out_path, semilog=True,
		show=show_plot)
	
	# Make weight plots
	title = 'Competitive Learning Network - Weights\n{0} Clusters'
	for weights, params, ncluster in zip(weight_results, weight_plot_params,
		nclusters):
		out_path = os.path.join(out_dir, 'weights-{0}.png'.format(ncluster))
		plot_weights(weights=weights, title=title.format(ncluster),
			out_path=out_path, show=show_plot, shape=(28, 28), **params)
	
	###########################################################################
	###### Vary boost increment and decrement
	###########################################################################
	
	print 'Varying boost increment and decrement amounts'
	space              = np.linspace(0.001, .1, 100)
	boost_pairs        = np.array([(x, y) for x in space for y in space])
	train_results      = np.zeros((len(boost_pairs), nepochs))
	train_stds         = np.zeros((len(boost_pairs), nepochs))
	test_results       = np.zeros((len(boost_pairs), nepochs))
	test_stds          = np.zeros((len(boost_pairs), nepochs))
	for i, pair in enumerate(boost_pairs):
		print 'Executing iteration {0} of {1}'.format(i + 1, len(boost_pairs))
		(train_results[i], train_stds[i]), (test_results[i], test_stds[i]) =  \
			bulk(train_data=train_d, test_data=test_d, plot=False,
			nepochs=nepochs, boost_inc=pair[0], boost_dec=pair[1],
			verbose=False, niters=niters)
	
	# Make training plot at last epoch
	title    = 'Competitive Learning Network - Training\n10 Iterations, '     \
		'Epoch {0}, Varying Boost Increment and Decrement'.format(nepochs)
	out_path = os.path.join(out_dir, 'boost-train.png')
	plot_surface(*make_grid(np.array([boost_pairs.T[0], boost_pairs.T[1],
		train_results.T[-1]]).T), x_label='Boost Increment', out_path=out_path,
		y_label='Boost Decrement', title=title, show=show_plot, z_label='Cost')
	
	# Make testing plot at last epoch
	title    = 'Competitive Learning Network - Testing\n10 Iterations, '      \
		'Epoch {0}, Varying Boost Increment and Decrement'.format(nepochs)
	out_path = os.path.join(out_dir, 'boost-test.png')
	plot_surface(*make_grid(np.array([boost_pairs.T[0], boost_pairs.T[1],
		test_results.T[-1]]).T), x_label='Boost Increment', out_path=out_path,
		y_label='Boost Decrement', title=title, show=show_plot, z_label='Cost')

if __name__ == '__main__':
	basic_sim()
	# bulk_sim()
	# vary_params(out_dir=r'D:\annt\test\Clustering_Network', show_plot=False)