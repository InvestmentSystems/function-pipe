from os import path

# To use a consistent encoding
from codecs import open
# Always prefer setuptools over distutils
from setuptools import setup

# https://packaging.python.org/distributing/
# to deploy:
# rm -r dist
# python setup.py sdist
# python setup.py bdist_wheel
# twine upload dist/*

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='function-pipe',
    version='2.0.0',

    description='Tools for extended function composition and pipelines',
    long_description=long_description,

    url='https://github.com/InvestmentSystems/function-pipe',
    author='Christopher Ariza, Charles Burkland',
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        ],

    keywords='functionnode pipenode composition pipeline pipe',
    py_modules=['function_pipe'], # no .py!
)
