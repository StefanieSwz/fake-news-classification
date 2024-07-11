# Continuous Integration

## Unit tests
Unit testing refers to the practice of writing test that tests individual parts of our code base to test for correctness. We use `Pytest` to test our code base. To locally check if our tests are passing, we type in a terminal `pytest tests/`. We measure the amount of code our tests cover with the package `coverage`. Then, we adapt the prompt to `coverage run -m pytest tests/`. To get a simple coverage report, type `coverage report` which will give us the percentage of cover in each of our files. By writing `coverage report -m`, we get the exact lines that were missed by our tests.

```bash
pytest tests/       # check if tests are passing
coverage report     # simple coverage report
coverage report -m  # exact lines missed by tests
```

## GitHub actions
Each repository gets 2000 minutes of free testing per month. It is added in `.github/workflows/tests.yaml` which should automatically run the tests for us when pushin to main or to a branch with PR to main. It works by initiating a Python environmen, installing all dependencies and then `pytest` is called to run the tests. We execute our tests on different operating systems, i.e. Linux, Windows and Mac. Our minutes for July are exhausted, therefore our tests fail right now when pushing on main, but work locally.

## Code style
We use `ruff` to check and re-format our code. The file `.github/workflows/codecheck.yaml` that also sets up a Python environment, installs the requirements and runs `ruff check .` and `ruff format .` which checks and formats the code according to certain Pep8 style guidelines, e.g. trimming white spacing.

## Pre-commit
Pre-commits can help us attach additional tasks that should be run every time that we do a `git commit`. Setup:

```bash
pip install pre-commit
pre-commit sample-config > .pre-commit-config.yaml #(or git pull, if file already exists, important: utf-8)
pre-commit install
pre-commit run --all-files  # fixes all files
git commit -m "Commit message" --no-verify  # usual commit
```

We can either add the modified files to the staging area and commit. If we have style issues for example, the pre-commit fixes those such that we can re-add them, commit and push. Otherwise we could run `pre-commit run --all-files` which fixes all files. If we want to do usual commits, we add as usual and then type `git commit -m "Commit message" --no-verify`.
