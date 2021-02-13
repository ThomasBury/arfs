import os.path
from setuptools import setup, find_packages

# The directory containing this file
HERE = os.path.abspath(os.path.dirname(__file__))

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

setup(name="arfs",
      version="0.1.4",
      description="All Relevant Feature Selection",
      long_description=README,
      long_description_content_type="text/markdown",
      url="https://github.com/ThomasBury/arfs",
      author="Thomas Bury",
      author_email='bury.thomas@gmail.com',
      packages=find_packages(),
      zip_safe=False,  # the package can run out of an .egg file
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      python_requires='>=3.6, <3.9',
      license='MIT',
      keywords=KEYWORDS
      )
