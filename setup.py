# To use a consistent encoding
from codecs import open
from os import path

# Always prefer setuptools over distutils
from setuptools import setup

# https://packaging.python.org/distributing/
# to deploy:
# rm -r dist
# python setup.py sdist
# python setup.py bdist_wheel
# twine upload dist/*

ROOT_DIR_FP = path.abspath(path.dirname(__file__))

def get_long_description() -> str:
    with open(path.join(ROOT_DIR_FP, "README.rst"), encoding="utf-8") as f:
        return f.read()

def get_version() -> str:
    with open(path.join(ROOT_DIR_FP, 'function_pipe', '__init__.py'),
            encoding='utf-8') as f:
        for l in f:
            if l.startswith('__version__'):
                if '#' in l:
                    l = l.split('#')[0].strip()
                return l.split('=')[-1].strip()[1:-1]
    raise ValueError('__version__ not found!')

setup(
    name="function-pipe",
    version=get_version(),

    description="Tools for extended function composition and pipelines",
    long_description=get_long_description(),

    url="https://github.com/InvestmentSystems/function-pipe",
    author="Christopher Ariza, Charles Burkland",
    license="MIT",

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        ],

    keywords="functionnode pipenode composition pipeline pipe",
    #py_modules=["function_pipe"], # no .py!
    packages=["function_pipe",
        "function_pipe.core",
        "function_pipe.test",
        ],
)
