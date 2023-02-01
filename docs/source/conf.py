# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'dynamiCXS'
copyright = '2023'
author = 'Nina Andrejevic'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
	'sphinx.ext.duration',
	'sphinx.ext.doctest',
	'sphinx.ext.autodoc',
	'sphinx.ext.autosummary',
	'sphinx.ext.intersphinx',
	'sphinx.ext.napoleon',
	'nbsphinx',
]

intersphinx_mapping = {
	'python': ('https://docs.python.org/3/', None),
	'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output
html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# -- Options for docstring
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True
add_module_names = False

# -- Options for nbsphinx
nbsphinx_execute = 'never'

import os
import sys
#Location of Sphinx files
sys.path.insert(0, os.path.abspath('./../..'))

# to display docs when using imported packages
def setup(app):
	import mock
	
	#MOCK_MODULES = ['numpy', 'matplotlib', 'matplotlib.pyplot', 'time', 'skimage.data', 'torchdiffeq', 'utils']
	MOCK_MODULES = []
	
	sys.modules.update((mod_name, mock.Mock()) for mod_name in MOCK_MODULES)

	#for mod_name in MOCK_MODULES:
    	#	sys.modules[mod_name] = mock.Mock()
    	#	sys.modules[mod_name].__name__ = mod_name

	from dynamicxs import ode
	ode.ODE.__name__ = 'ODE'
	ode.Kuramoto.__name__ = 'Kuramoto'
	ode.GrayScott.__name__ = 'GrayScott'
	ode.LotkaVolterra.__name__ = 'LotkaVolterra'
	sys.modules['ode'] = ode

	app.connect('build-finished', build_finished)

def build_finished(app, exception):
	pass

