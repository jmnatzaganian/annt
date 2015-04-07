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
from annt.util                      import flip_random_bits
from annt.experimental.hopfield_net import Hopfield
from annt.plot                      import plot_weights

def main(train_data, train_labels, nsamples=10, nstates=1, labels=[0],
	pct_noise=0.3, plot=True):
	"""
	Demonstrates a hopfield network using MNIST.
	
	@param train_data: The data to train with. This must be an iterable
	returning a numpy array.
	
	@param train_labels: The training labels. This must be an iterable with the
	same length as train_data.
	
	@param nsamples: The number of samples to use for generating the attractor
	states.
	
	@param nstates: The number of states to create. This will result in
	creating a number of attractor states equal to the amount of unique labels
	multiplied by this value.
	
	@param labels: This parameter should be a list of the labels to use. If it
	is None then all unique labels will be used.
	
	@param pct_noise: The percentage of noise to be added to the attractor
	state.
	
	@param plot: If True one or more plots are generated showing the attractor
	states.
	
	@return: A list of lists containing the attractor state, the attractor
	state with noise, and the found attractor state, respectively.
	"""
	
	# Create the network
	net = Hopfield()
	
	# Initialize the attractor states
	net.create_attractor_states(train_data, train_labels, nstates, nsamples,
		255 / 2, labels=labels)
	
	# Check noise tolerance with all of the activation states
	all_states = []
	for state in net.attractor_states.keys():
		# Get the states
		states = []
		states.append(np.array(state))
		states.append(flip_random_bits(states[0], pct_noise))
		states.append(net.step(states[1]))
		all_states.append(np.copy(states))
		
		# Make the plot
		if plot:
			plot_weights(states, 1, 3, (28, 28), title='Hopfield Network - '
				'Noise Tolerance Example\nFlipped {0}% of the bits'.format(
				pct_noise * 100), cluster_titles=('Attractor State',
				'Attractor State with Noise', 'Found Attractor'))
	
	return all_states

def basic_sim():
	"""
	Perform a basic simulation.
	"""
	
	# Get the data
	(train_data, train_labels), (test_data, test_labels) = mnist_data()
	
	# Run the network
	main(threshold(train_data, 255 / 2), train_labels)

def vary_params(out_dir, show_plot=True, seed=None):
	"""
	Vary some parameters and generate some plots.
	
	@param out_dir: The directory to save the plots in.
	
	@param show_plot: If True the plot will be displayed upon creation.
	
	@param seed: The seed for the random number generator. This will force the
	network to work with the same data.
	"""	
	
	# Get the data
	(train_data, train_labels), (test_data, test_labels) = mnist_data()
	train_d = threshold(train_data, 255 / 2)
	
	# Make the output directory
	try:
		os.makedirs(out_dir)
	except OSError:
		pass
	
	###########################################################################
	###### Vary number of attractors
	###########################################################################
	
	cluster_titles = ['Attractor State', 'Input', 'Found Attractor']
	for i in xrange(1, 4):
		np.random.seed(seed)
		states = np.array(main(train_d, train_labels, labels=np.arange(i),
			pct_noise=0, plot=False))
		plot_weights(states.reshape(states.shape[1] * states.shape[0],
			states.shape[2]), i, 3, (28, 28), title='Hopfield Network - {0} '
			'Attractor State(s)'.format(i), cluster_titles=cluster_titles * i,
			out_path=os.path.join(out_dir, 'states_{0}.png'.format(i)),
			show=show_plot)
	
	###########################################################################
	###### Vary amount of noise on input
	###########################################################################
	
	noise_ranges = (0.4, 0.6)
	cluster_titles = ['Attractor State', 'Input with {0}% Noise',
		'Found Attractor'] * len(noise_ranges)
	for i, noise in enumerate(noise_ranges):
		cluster_titles[i * 3 + 1] = cluster_titles[i * 3 + 1].format(noise *
			100)
	for i in xrange(1, 3):
		new_title = [y for x in [[cluster_titles[0], cluster_titles[3 * j + 1],
			cluster_titles[2]] * i for j in xrange(len(noise_ranges))]
			for y in x]
		states = []
		for noise in noise_ranges:
			np.random.seed(seed)
			states.extend(main(train_d, train_labels, labels=np.arange(i),
				pct_noise=noise, plot=False))
		states = np.array(states)
		plot_weights(states.reshape(states.shape[1] * states.shape[0],
			states.shape[2]), len(noise_ranges) * i, 3, (28, 28),
			title='Hopfield Network - Noise  Tolerance'.format(noise * 100),
			cluster_titles=new_title, out_path=os.path.join(out_dir,
			'noise_states_{0}.png'.format(i)), show=show_plot)

if __name__ == '__main__':
	basic_sim()
	# vary_params(out_dir=r'D:\annt\test\Hopfield_Network', show_plot=False,
		# seed=123456789)