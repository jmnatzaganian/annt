# plot.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Date Created   : 03/31/15
#	
# Description    : Module for plotting.
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
Module for plotting.

G{packagetree annt}
"""

__docformat__ = 'epytext'

# Native imports
import itertools

# Third-Party imports
import numpy             as np
import matplotlib.pyplot as plt

def plot_epoch(y_series, series_names=None, y_errs=None, y_label=None,
	title=None, semilog=False, legend_location='best', out_path=None,
	show=True):
	"""
	Basic plotter function for plotting various types of data against
	training epochs. Each item in the series should correspond to a single
	data point for that epoch.
	
	@param y_series: A tuple containing all of the desired series to plot.
	
	@param series_names: A tuple containing the names of the series.
	
	@param y_errs: The error in the y values. There should be one per series
	per datapoint. It is assumed this is the standard deviation, but any error
	will work.
	
	@param y_label: The label to use for the y-axis.
	
	@param title: The name of the plot.
	
	@param semilog: If True the y-axis will be plotted using a log(10) scale.
	
	@param legend_location: The location of where the legend should be placed.
	Refer to matplotlib's U{docs<http://matplotlib.org/api/pyplot_api.html#
	matplotlib.pyplot.legend>} for more details.
	
	@param out_path: The full path to where the image should be saved. The file
	extension of this path will be used as the format type. If this value is
	None then the plot will not be saved, but displayed only.
	
	@param show: If True the plot will be show upon creation.
	"""
	
	# Construct the basic plot
	fig, ax = plt.subplots()
	if title is not None   : plt.title(title)
	if semilog             : ax.set_yscale('log')
	if y_label is not None : ax.set_ylabel(y_label)
	ax.set_xlabel('Epoch')
	plt.xlim((1, max([x.shape[0] for x in y_series])))
	colormap = plt.cm.brg
	colors   = itertools.cycle([colormap(i) for i in np.linspace(0, 0.9,
		len(y_series))])
	markers  = itertools.cycle(['.', ',', 'o', 'v', '^', '<', '>', '1', '2',
		'3', '4', '8', 's', 'p', '*', 'p', 'h', 'H', '+', 'D', 'd', '|', '_',
		'TICKLEFT', 'TICKRIGHT', 'TICKUP', 'TICKDOWN', 'CARETLEFT',
		'CARETRIGHT', 'CARETUP', 'CARETDOWN'])
	
	# Add the data
	if y_errs is not None:
		for y, err in zip(y_series, y_errs):
			x = np.arange(1, x.shape[0] + 1)
			ax.errorbar(x, y, yerr=err, color=colors.next(),
				marker=markers.next())
	else:
		for y in y_series:
			x = np.arange(1, x.shape[0] + 1)
			ax.scatter(x, y, color=colors.next(), marker=markers.next())
	
	# Create the legend
	if series_names is not None: plt.legend(series_names, loc=legend_location)
	
	# Save the plot
	fig.set_size_inches(19.20, 10.80)
	if out_path is not None:
		plt.savefig(out_path, format=out_path.split('.')[-1], dpi = 100)
	
	# Show the plot and close it after the user is done
	if show: plt.show()
	plt.close()

def plot_weights(weights, nrows, ncols, shape, title=None, out_path=None,
	show=True):
	"""
	Plot the weight matrices for the network.
	
	@param weights: A numpy array containing a weight matrix. Each row in the
	array corresponds to a unique node. Each column corresponds to a weight
	value.
	
	@param nrows: The number of rows of plots to create.
	
	@param ncols: The number of columns of plots to create.
	
	@param shape: The shape of the weights. It is assumed that a 1D shape was
	used and is desired to be represented in 2D. Whatever shape is provided
	will be used to reshape the weights. For example, if you had a 28x28 image
	and each weight corresponded to one pixel, you would have a vector with a
	shape of (784, ). This vector would then need to be resized to your desired
	shape of (28, 28).
	
	@param title: The name of the plot.
	
	@param out_path: The full path to where the image should be saved. The file
	extension of this path will be used as the format type. If this value is
	None then the plot will not be saved, but displayed only.
	
	@param show: If True the plot will be show upon creation.
	"""
	
	# Construct the basic plot
	fig = plt.figure()
	if title is not None: fig.suptitle(title, fontsize=16)
	
	# Add all of the figures to the grid
	for i, weight_set in enumerate(weights):
		ax = plt.subplot(nrows, ncols, i + 1)
		ax.set_title('Node {0}'.format(i))
		ax.imshow(weight_set.reshape(shape), cmap=plt.cm.gray)
		ax.axes.get_xaxis().set_visible(False)
		ax.axes.get_yaxis().set_visible(False)
	
	# Save the plot
	fig.set_size_inches(19.20, 10.80)
	if out_path is not None:
		plt.savefig(out_path, format=out_path.split('.')[-1], dpi = 100)
	
	# Show the plot and close it after the user is done
	if show is not None: plt.show()
	plt.close()