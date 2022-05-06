import sys

import invoke

#-------------------------------------------------------------------------------

@invoke.task
def clean(context):
    """Clean doc and build artifacts
    """
    context.run("rm -rf coverage.xml")
    context.run("rm -rf htmlcov")
    context.run("rm -rf doc/build")
    context.run("rm -rf build")
    context.run("rm -rf dist")
    context.run("rm -rf *.egg-info")
    context.run("rm -rf .coverage")
    context.run("rm -rf .mypy_cache")
    context.run("rm -rf .pytest_cache")
    context.run("rm -rf .hypothesis")
    context.run("rm -rf .ipynb_checkpoints")



@invoke.task()
def doc(context):
    """Build docs
    """
    context.run(f"{sys.executable} doc/doc_build.py")


@invoke.task
def test(context, cov=False):
    """Run tests.
    """
    cmd = f"pytest -s function_pipe/test"

    if cov:
        cmd += " --cov=function_pipe --cov-report=xml"

    print(cmd)
    context.run(cmd)


@invoke.task
def coverage(context):
    """
    Perform code coverage, and open report HTML.
    """
    cmd = "pytest -s --color no --cov=function_pipe/core --cov-report html"
    print(cmd)
    context.run(cmd)
    import webbrowser
    webbrowser.open("htmlcov/index.html")


@invoke.task
def mypy(context):
    """Run mypy static analysis.
    """
    context.run("mypy --strict")


@invoke.task
def lint(context):
    """Run pylint static analysis.
    """
    context.run("pylint -f colorized function_pipe")

@invoke.task(pre=(mypy, lint))
def quality(context):
    """Perform all quality checks.
    """


@invoke.task
def isort(context, tlt=None, check=False):
    """Sort imports."""
    cmd = "isort"
    if check:
        cmd += " --check"
    context.run(cmd, echo=True, pty=True)


@invoke.task
def black(context, tlt=None, check=False):
    """Format code."""
    args = ["black"]
    if check:
        args.append("--check")
    if tlt is not None:
        args.append(expand_tlt_package(tlt))
    else:
        args.extend(get_targets())
    command = quote_and_join(args)
    context.run(command, echo=True, pty=True)

@invoke.task  # Don't put the jobs here, since we need to forward args to them!
def formatting(context, tlt=None, check=False, default=False):
    """Run all formatting checks."""
    if default and VcsUtil.get_repo_branch() != VcsUtil.DEFAULT_BRANCH:
        return
    # Fastest to slowest:
    init(context, tlt=tlt, check=check)
    black(context, tlt=tlt, check=check)
    isort(context, tlt=tlt, check=check)


#-------------------------------------------------------------------------------

@invoke.task(pre=(clean,))
def build(context):
    """Build packages
    """
    context.run(f"{sys.executable} setup.py sdist bdist_wheel")

@invoke.task(pre=(build,), post=(clean,))
def release(context):
    context.run("twine upload dist/*")
