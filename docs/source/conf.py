# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'neuroptimiser'
copyright = '2025, Jorge M. Cruz-Duarte (jorge-mario.cruz-duarte@inria.fr)'
author = 'Jorge M. Cruz-Duarte'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

autosummary_generate = True
# html_theme = 'sphinx_rtd_theme'
# html_logo = "_pics/neuropty.png"
html_theme = "furo"
html_title = "Neuroptimiser"
html_favicon = '_static/neuropty-Light-small.png'
html_static_path = ["_static"]
# html_css_files = ['custom.css']
html_extra_path = ['.nojekyll']
html_baseurl = "https://jcrvz.github.io/neuroptimiser/"

html_theme_options = {
    "navigation_with_keys": True,
    "announcement": "This is a WIP project, and the documentation is under construction.",
    "body_max_width": "100%",
    "light_logo": "neuropty-Light2x.png",
    "dark_logo": "neuropty-Dark2x.png",
    "light_css_variables": {
        "color-brand-primary": "#3B4CCA",    # sober blue accent
        "color-brand-content": "#1A1A1A",    # dark grey for text
        "color-background-primary": "#FFFFFF",  # pure white background
        "color-background-secondary": "#F7F7F7",  # very light grey for sections
        "color-sidebar-background": "#F0F0F0",    # light grey sidebar
        "color-sidebar-link-text": "#333333",  # almost black sidebar links
        "color-sidebar-link-text--top-level": "#1A1A1A",
        "color-sidebar-link-background--active": "#E6E6E6",
    },
    "dark_css_variables": {
        "color-brand-primary": "#7DA2FF",    # soft blue for dark mode
        "color-brand-content": "#E0E0E0",    # light grey text
        "color-background-primary": "#181818",  # dark background
        "color-background-secondary": "#242424",  # slightly lighter sections
        "color-sidebar-background": "#202020",
        "color-sidebar-link-text": "#CCCCCC",
        "color-sidebar-link-text--top-level": "#FFFFFF",
        "color-sidebar-link-background--active": "#333333",  # active link background
    }
}