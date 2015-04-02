# setup.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Date Created   : 03/30/15
#	
# Description    : Installs the annt project
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

# Native imports
from distutils.core import setup
import shutil

# Install the program
setup(
	name='annt',
	version='0.3.0',
	description="Artificial Neural Network Toolbox",
	author='James Mnatzaganian',
	author_email='jamesmnatzaganian@outlook.com',
	url='http://techtorials.me',
	packages=['annt', 'annt.examples'],
	package_data={'annt.examples':['data/mnist.pkl']}
	)

# Remove the unnecessary build folder
try:
	shutil.rmtree('build')
except:
	pass