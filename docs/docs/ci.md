# Continuous Integration

## Unit tests
Unit testing refers to the practice of writing test that tests individual parts of our code base to test for correctness. We use `Pytest` to test our code base. To locally check if our tests are passing, we type in a terminal `pytest tests/`. We measure the amount of code our tests cover with the package `coverage`. Then, we adapt the prompt to `coverage run -m pytest tests/`. To get a simple coverage report, type `coverage report` which will give us the percentage of cover in each of our files. By writing `coverage report -m`, we get the exact lines that were missed by our tests.
