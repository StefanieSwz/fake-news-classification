# Continuous Integration

## Unit tests

### Important commands
Unit testing refers to the practice of writing test that tests individual parts of our code base to test for correctness. We use `Pytest` to test our code base. To locally check if our tests are passing, we type in a terminal `pytest tests/`. We measure the amount of code our tests cover with the package `coverage`. Then, we adapt the prompt to `coverage run -m pytest tests/`. To get a simple coverage report, type `coverage report` which will give us the percentage of cover in each of our files. By writing `coverage report -m`, we get the exact lines that were missed by our tests.

```bash
pytest tests/                    # check if tests are passing
```
```bash
coverage run -m pytest tests/    # alternative, checks also coverage
coverage report                  # simple coverage report
coverage report -m               # exact lines missed by tests
```

### Files to test
In total we have implemented **39 tests**. We are testing the data part of the project, especially with `make_dataset.py` and `preprocessing.py`, and the model related part, which is aiming at the `model.py` and `train_model.py`. The other scripts of the module are mainly consisting of the one function, which is highly entangeled with other services like wandb. The testing is more complex here, since this functionality has to be mocked and can't be tested. Additionally, we wrote two test files for integration test of the inference and monitoring app. We also allow load testing for inference backend using locust.

The total code coverage of our code is 75%, which includes `fakenews\config.py`, `fakenews\data\make_dataset.py`, `fakenews\data\preprocessing.py`, `fakenews\model\model.py`, `fakenews\model\train_model.py`, and all related init and test files. While we strive for higher coverage, achieving 100% coverage is challenging due to the complexity and interactions of advanced functions, many of which are wrapped within other functions, making them difficult to test in isolation.

Even if we were to reach 100% code coverage, it would not necessarily guarantee that the code is error-free. Code coverage metrics indicate how much of the code is executed during testing, but they do not guarantee the absence of logical errors, edge cases, or unforeseen interactions. It is crucial to complement high code coverage with comprehensive testing strategies, including integration tests, system tests, and manual testing, to ensure the robustness and reliability of the software.

## GitHub actions
Each repository gets 2000 minutes of free testing per month. It is added in `.github/workflows/tests.yaml` which should automatically run the tests for us when pushing to main or to a branch with PR to main. It works by initiating a Python environment, installing all dependencies and then `pytest` is called to run the tests. We execute our tests on different operating systems, i.e. Linux, Windows and Mac. Our minutes for July are exhausted, therefore our tests fail right now when pushing on main, but work locally.

## Code style
We use `ruff` to check and re-format our code. The file `.github/workflows/codecheck.yaml` that also sets up a Python environment, installs the requirements and runs `ruff check .` and `ruff format .` which checks and formats the code according to certain Pep8 style guidelines, e.g. trimming white spacing.
To fix error outputs type `ruff check . --fix`.

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
