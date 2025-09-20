import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "edges"
copyright = "2025"
author = "Paul Scherrer Institute"

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
templates_path = ["_templates"]
exclude_patterns = []


html_theme = "alabaster"
html_static_path = ["_static"]
html_logo = "https://raw.githubusercontent.com/Laboratory-for-Energy-Systems-Analysis/edges/refs/heads/main/assets/permanent/edges_logo_tight_frame.png"

import os
import sys

sys.path.insert(0, os.path.abspath("../"))  # or '../src' if your code is in src/
