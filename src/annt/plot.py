# plot.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Organization   : NanoComputing Research Lab - Rochester Institute of
# Technology
# http://www.rit.edu/kgcoe/facility/nanocomputing-research-laboratory
# http://nano.ce.rit.edu
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
import os, itertools

# Third-Party imports
import numpy             as np
import matplotlib.pyplot as plt
from   matplotlib.ticker import MultipleLocator

def basic_epoch(y_series, series_names=None, y_label=None, title=None,
	out_path=None, show=True):
	"""
	Basic plotter function for plotting various types of data against
	training epochs. Each item in the series should correspond to a single
	data point for that epoch.
	
	@param y_series: A tuple containing all of the desired series to plot.
	
	@param series_names: A tuple containing the names of the series.
	
	@param y_label: The label to use for the y-axis.
	
	@param title: The name of the plot.
	
	@param out_path: The full path to where the image should be saved. The file
	extension of this path will be used as the format type. If this value is
	None then the plot will not be saved, but displayed only.
	
	@param show: If True the plot will be show upon creation.
	"""
	
	# Construct the basic plot
	fig, ax = plt.subplots()
	if title is not None   : plt.title(title)
	ax.set_xlabel('Epoch')
	plt.xlim((1, max([x.shape[0] for x in y_series])))
	ax.set_yscale('log')
	if y_label is not None : ax.set_ylabel(y_label)
	colormap = plt.cm.brg
	colors   = itertools.cycle([colormap(i) for i in np.linspace(0, 0.9,
		len(y_series))])
	markers  = itertools.cycle(['.', ',', 'o', 'v', '^', '<', '>', '1', '2',
		'3', '4', '8', 's', 'p', '*', 'p', 'h', 'H', '+', 'D', 'd', '|', '_',
		'TICKLEFT', 'TICKRIGHT', 'TICKUP', 'TICKDOWN', 'CARETLEFT',
		'CARETRIGHT', 'CARETUP', 'CARETDOWN'])
		
	# Add the data
	for y in y_series:
		x = np.arange(1, x.shape[0] + 1)
		ax.scatter(x, y, color=colors.next(), marker=markers.next())
	
	# Create the legend
	if series_names is not None: plt.legend(series_names)
	
	# Save the plot
	if out_path is not None:
		plt.savefig(out_path, format=out_path.split('.')[-1])
	
	# Show the plot and close it after the user is done
	if show is not None: plt.show()
	plt.close()