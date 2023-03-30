import os
import sys
sys.path.insert(0, os.path.abspath('../../GGanalysis'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GGanalysis'
copyright = '2023, OneBST'
author = 'OneBST'
release = '0.3.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.githubpages',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx_copybutton',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'zh_CN'
html_search_language = 'zh'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo" # pip install furo
html_static_path = ['_static']
html_css_files = ['custom.css']
html_title = "GGanalysis"
html_theme_options = {
    "sidebar_hide_name": False,  # 启用左上角的标题以便随时返回
}