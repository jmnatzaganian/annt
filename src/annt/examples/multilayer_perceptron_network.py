# multilayer_perceptron_network.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Date Created   : 03/31/15
#	
# Description    : Example showing how to create and use a multilayer
# perceptron network.
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
Example showing how to create and use a multilayer perceptron network. This
example uses a reduced set of data from the
U{MNIST<http://yann.lecun.com/exdb/mnist/>} dataset.

G{packagetree annt}
"""

__docformat__ = 'epytext'

# Program imports
from annt.util import one_hot, mnist_data
from annt.net  import MultilayerPerception
from annt.plot import plot_epoch

def main(train_data, train_labels, test_data, test_labels, nepochs=1, bias=1,
	learning_rate=0.001, plot=True, verbose=True, min_weight=-1, max_weight=1,
	hidden_activation_type='sigmoid', activation_kargs={'m':1}):
	"""
	Demonstrates a multilayer perceptron network using MNIST.
	
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
	
	@param hidden_activation_type: The type activation function to use for the
	hidden neurons. This must be one of the classes implemented in
	L{annt.activation}.
	
	@param activation_kargs: Any keyword arguments for the activation
	function.
	"""
	
	# Create the network
	net =  MultilayerPerception(
		shape                  = [train_data.shape[1], 100, 10],
		bias                   = bias,
		learning_rate          = learning_rate,
		min_weight             = min_weight,
		max_weight             = max_weight,
		hidden_activation_type = hidden_activation_type
	)
	
	# Simulate the network
	train_results, test_results = net.run(train_data, train_labels,
		test_data, test_labels, nepochs, verbose)
	
	# Plot the results
	if plot:
		plot_epoch(y_series=(train_results * 100, test_results * 100),
			series_names=('Train', 'Test'), y_label='Accuracy [%]',
			title='MLP - Example', legend_location='upper left')
	
	return train_results * 100, test_results * 100

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
	main(train_data/255., one_hot(train_labels, 10), test_data/255.,
		one_hot(test_labels, 10), nepochs=100)

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
		plot_epoch(y_series=(train_mean, test_mean),
			legend_location='upper left'
			series_names=('Train', 'Test'), y_errs=(train_std, test_std),
			y_label='Accuracy [%]',	title='Linear Regression Network - Stats Example',
			out_path=out_path)
	
	return (train_mean, train_std), (test_mean, test_std)

if __name__ == '__main__':
	basic_sim(10)