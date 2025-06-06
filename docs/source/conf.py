# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('../src'))

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
    "myst_nb",
]
add_module_names = False
nb_execution_mode = "off"

autoclass_content = 'class'
autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
autodoc_class_signature = 'separated'
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'sphinx_rtd_theme'
# html_logo = "_pics/neuropty.png"
html_theme = "furo"
html_title = "NeurOptimiser"
html_favicon = '_static/neuropty-Light-small.png'
html_static_path = ["_static"]
html_css_files = ['custom.css']
html_extra_path = ['.nojekyll']
html_baseurl = "https://neuroptimiser.github.io/"
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",  # if using ReadTheDocs Ads
    ]
}

html_theme_options = {
    "navigation_with_keys": True,
    "announcement": "This is a WIP project, and the documentation is under construction.",
    "body_max_width": "100%",
    "light_logo": "neuropty-Light-small.png",
    "dark_logo": "neuropty-Dark-small.png",
    "light_css_variables": {
        "color-brand-primary": "#A5B68D",    # sober blue accent
        "color-brand-content": "#1A1A1A",    # dark grey for text
        "color-background-primary": "#FFFFFE",  # quite-white background
        "color-background-secondary": "#F7F7F7",  # very light grey for sections
        "color-sidebar-background": "#E4E0E1",    # light grey sidebar
        "color-sidebar-link-text": "#333333",  # almost black sidebar links
        "color-sidebar-link-text--top-level": "#1A1A1A",
        "color-sidebar-link-background--active": "#E6E6E6",
        "sidebar-width": "50px",
        "sidebar-width--mobile": "70vw",
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
        "sidebar-width": "50px",
        "sidebar-width--mobile": "70vw",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/jcrvz/neuroptimiser",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0"
                     viewBox="0 0 1024 1024" height="1.2em" width="1.2em"
                     xmlns="http://www.w3.org/2000/svg">
                    <path d="M511.6 76C262.2 76 64 274.2 64 523.6c0 197.6 128 365 305.4 424.2 22.4 4.2 30.6-9.8 30.6-21.8v-76.2c-124.2 27-150.2-59.8-150.2-59.8-20.4-52-49.8-65.8-49.8-65.8-40.8-27.8 3-27.2 3-27.2 45 3.2 68.6 46.2 68.6 46.2 40 68.4 104.8 48.6 130.4 37.2 4-29 15.6-48.6 28.4-59.8-99-11.2-202.8-49.6-202.8-220.6 0-48.6 17.2-88.4 45.4-119.6-4.6-11.2-19.6-56.4 4.4-117.4 0 0 37-11.8 121.2 45 35.2-9.8 72.8-14.8 110.2-14.8s75 5 110.2 14.8c84.2-56.8 121.2-45 121.2-45 24 61 9 106.2 4.4 117.4 28.4 31.2 45.4 71 45.4 119.6 0 171.6-103.8 209.4-202.8 220.4 16 13.8 30.4 41 30.4 82.6v122.2c0 12 8 26.2 30.8 21.8C832 888.4 960 720.8 960 523.6 960 274.2 761.8 76 511.6 76z"></path>
                </svg>
            """,
            "class": "",  # optional css class
        },
    ],
}

napoleon_use_param = True
napoleon_use_rtype = True
nb_remove_code_source = True
