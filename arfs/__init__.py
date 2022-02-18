"""init module, providing information about the arfs package
"""

import os.path

# The directory containing this file
HERE = os.path.abspath(os.path.dirname(__file__))

# get key package details from taco/__version__.py
ABOUT = {}  # type: ignore
with open(os.path.join(HERE, '__version__.py')) as f:
    exec(f.read(), ABOUT)

__version__ = ABOUT['__version__']
