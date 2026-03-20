import os
import sys
  # añade la raíz del proyecto
sys.path.insert(0, os.path.abspath('..'))
# -- Project information -----------------------------------------------------

project = 'Pixel Neural Networks'
author = 'Tu Nombre'
release = '1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',    # genera documentación a partir de docstrings
    'sphinx.ext.napoleon',   # soporta Google / NumPy style docstrings
    'sphinx.ext.viewcode',   # enlaces al código fuente
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']