# __init__.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Date Created   : 03/30/15
#
# Description    : Defines the annt package
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
This is a collection of various components for creating artificial neural
networks in Python.

Legal
=====
	This code is licensed under the U{MIT license<http://opensource.org/
	licenses/mit-license.php>}. Any included datasets may be licensed
	differently. Refer to the individual dataset for more details.

Prerequisites
=============
	- U{Python 2.7.X<https://www.python.org/downloads/release/python-279/>}
	- U{Numpy<http://www.numpy.org/>}
	- U{matplotlib<http://matplotlib.org/>}

Installation
============
	1. Install all prerequisites
		Assuming you have U{pip<https://pip.pypa.io/en/latest/installing.html>}
		installed, located in your X{Python27/Scripts} directory:
		
		X{pip install numpy matplotlib}
	2. Install this package: X{python setup.py install}. The setup file is
	located in the "src" folder.

Getting Started
===============
	WIP

Package Organization
====================
	The annt package contains the following subpackages and modules:

	G{packagetree annt}

Connectivity
============
	The following image shows how everything is connected:

	G{importgraph}

Developer Notes
===============
	The following notes are for developers only.

	Installation
	------------
		1.  Download and install U{graphviz<http://www.graphviz.org/Download..
		php>}
		2.  Edit line 111 in X{dev/epydoc_config.txt} to point to the directory
		containing "dot.exe". This is part of the graphviz installation.
		3.  Download this repo and execute X{python setup.py install}.
		4.  Download and install U{Epydoc<http://sourceforge.net/projects/
		epydoc/files>}

	Generating the API
	------------------
		From the root level, execute X{python epydoc --config=epydoc_config.txt
		annt}

@group Examples: examples

@author: U{James Mnatzaganian<http://techtorials.me>}
@requires: Python 2.7.X
@version: 0.3.0
@license: U{The MIT License<http://opensource.org/licenses/mit-license.php>}
@copyright: S{copy} 2015 James Mnatzaganian
"""

__docformat__ = 'epytext'