#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# LMChallenge documentation build configuration file, created by
# sphinx-quickstart on Thu Oct 29 11:11:46 2015.

import sys
import os
import shlex
import recommonmark.parser

sys.path.insert(0, '..')

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinxarg.ext',
]
templates_path = ['_templates']
source_suffix = ['.rst', '.md']
source_parsers = {
    '.md': recommonmark.parser.CommonMarkParser,
}

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'LMChallenge'
copyright = '2015, SwiftKey'
author = 'SwiftKey'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
with open('../version.txt','r') as v:
    release = v.readline().strip()
version = release.split('.')[0]

language = None
exclude_patterns = ['_build']
pygments_style = 'sphinx'

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
htmlhelp_basename = 'LMChallengedoc'

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'lmchallenge', 'LMChallenge Documentation', [author], 1),
    ('wp', 'lmchallenge.wp', 'LMChallenge Word Prediction Documentation', [author], 1),
    ('tc', 'lmchallenge.tc', 'LMChallenge Word Prediction Documentation', [author], 1),
]
