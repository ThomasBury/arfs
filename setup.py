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

INSTALL_REQUIRES = ['pandas >= 1.0.0',
                    'numpy >= 1.18.0',
                    'scipy >= 1.4.0',
                    'scikit-learn >= 0.23.0',
                    'lightgbm >= 3.0.0',
                    'matplotlib >= 3.3.0',
                    'palettable >= 3.3.0',
                    'holoviews >= 1.13.0',
                    'shap >= 0.35.0',
                    'tqdm >= 4.40.0']

KEYWORDS = 'feature-selection, all-relevant, selection'

setup(name="arfs",
      version="0.1.2",
      description="All Relevant Feature Selection",
      long_description=README,
      long_description_content_type="text/markdown",
      url="https://github.com/ThomasBury/arfs",
      author="Thomas Bury",
      author_email='thomas.bury@gmail.com',
      packages=find_packages(),
      zip_safe=False,  # the package can run out of an .egg file
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      python_requires='>=3.6',
      license='MIT',
      keywords=KEYWORDS
      )
