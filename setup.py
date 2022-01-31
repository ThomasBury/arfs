from setuptools import setup, find_packages
import os.path

# The directory containing this file
HERE = os.path.abspath(os.path.dirname(__file__))

# get key package details from taco/__version__.py
ABOUT = {}  # type: ignore
with open(os.path.join(HERE, 'arfs', '__version__.py')) as f:
    exec(f.read(), ABOUT)

# The text of the README file
with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()

EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov']
}

# Get the requirements list by reading the file and splitting it up
with open('requirements.txt', 'r') as f:
    INSTALL_REQUIRES = f.read().splitlines()

KEYWORDS = 'feature-selection, all-relevant, selection'

setup(name=ABOUT['__title__'],
      version=ABOUT['__version__'],
      description=ABOUT['__description__'],
      long_description=README,
      long_description_content_type="text/markdown",
      url=ABOUT['__url__'],
      author=ABOUT['__author__'],
      author_email=ABOUT['__author_email__'],
      packages=find_packages(),
      zip_safe=False,  # the package can run out of an .egg file
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      python_requires='>=3.6',
      license=ABOUT['__license__'],
      keywords=KEYWORDS,
      package_data={'': ['datasets/data/*.zip',
                         'datasets/descr/*.rst']},
      )
