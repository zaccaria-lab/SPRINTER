# Basic requirements

SPRINTER is written in Python 3, requires version >= 3.9 of Python, and depends on the following standard Python packages, which must be available in the python environment where the user runs SPRINTER.

| Package | Tested version with Python 3.10.15 | Required |
|---------|----------------|----------|
| hmmlearn | 0.3.0 | >= 0.2.7 |
| matplotlib-base | 3.8.2 | |
| numba | 0.60.0 | |
| numpy | 1.26.2 | !=, < 2.* |
| pandas | 2.1.0 | 2.2.3 |
| pybedtools | 0.10.0 | |
| scikit-learn | 1.5.2 | |
| scipy | 1.14.1 | |
| seaborn | 0.13.2 | |
| statsmodels | 0.14.4 | |

If all these packages are correctly installed in the available Python environment, SPRINTER can be executed and it can be installed with `setuptools` through `pip` by running the following command from the root of this repository.
```shell
pip install .
```
