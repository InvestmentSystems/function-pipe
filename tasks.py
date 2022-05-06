import sys

import invoke

#-------------------------------------------------------------------------------

@invoke.task
def clean(context):
    '''Clean doc and build artifacts
    '''
    context.run('rm -rf coverage.xml')
    context.run('rm -rf htmlcov')
    context.run('rm -rf doc/build')
    context.run('rm -rf build')
    context.run('rm -rf dist')
    context.run('rm -rf *.egg-info')
    context.run('rm -rf .coverage')
    context.run('rm -rf .mypy_cache')
    context.run('rm -rf .pytest_cache')
    context.run('rm -rf .hypothesis')
    context.run('rm -rf .ipynb_checkpoints')



@invoke.task()
def doc(context):
    '''Build docs
    '''
    context.run(f'{sys.executable} doc/doc_build.py')


@invoke.task
def test(context, cov=False):
    '''Run tests.
    '''
    cmd = f'pytest -s --color no --tb=native test_function_pipe.py'

    if cov:
        cmd += ' --cov=function_pipe --cov-report=xml'

    print(cmd)
    context.run(cmd)


@invoke.task
def coverage(context):
    '''
    Perform code coverage, and open report HTML.
    '''
    cmd = 'pytest -s --color no --cov=function_pipe --cov-report html'
    print(cmd)
    context.run(cmd)
    import webbrowser
    webbrowser.open('htmlcov/index.html')


@invoke.task
def mypy(context):
    '''Run mypy static analysis.
    '''
    context.run('mypy --strict')


@invoke.task
def lint(context):
    '''Run pylint static analysis.
    '''
    context.run('pylint -f colorized function_pipe')

@invoke.task(pre=(mypy, lint))
def quality(context):
    '''Perform all quality checks.
    '''

#-------------------------------------------------------------------------------

@invoke.task(pre=(clean,))
def build(context):
    '''Build packages
    '''
    context.run(f'{sys.executable} setup.py sdist bdist_wheel')

@invoke.task(pre=(build,), post=(clean,))
def release(context):
    context.run('twine upload dist/*')


