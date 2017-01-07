# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

long_description = 'TODO'

setup(
    name='function-pipe',
    version='1.0.0',

    description='Tools for extended function composition and pipelines',
    long_description=long_description,

    url='https://github.com/InvestmentSystems/function-pipe',
    author='Christopher Ariza',
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
