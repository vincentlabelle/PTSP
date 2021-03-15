# PTSP (Power Two Samples Permutation)

This is a translation of an R function to calculate the power of a one-sided
permutation test on unpaired samples with equal variance and an equal number of
observations. The samples follow a normal distribution.

In this translation the above mentioned function was converted into a Python
object, ie. `UnpairedOneSidedPermutationTestPowerSimulator`.

The code in this package includes a lot of utilities that would be useful in a
statistical library. These utilities were separated from the main object to
perform the calculation of power for multiple reasons:

    1. Facilitate reuse
    2. Improve maintainability
    3. Improve scalability of the code

## Code Structure

### core/

Includes the statistical utilities and the object performing the power
calculation, ie. `UnpairedOneSidedPermutationTestPowerSimulator`.

Specifically, *core/core.py* includes the top-level object to perform the power
calculation.

The remaining files are utilities:

* *core/permutation.py* includes utilities to perform permutation tests and
  calculate the p-value of such tests.
* *core/random.py* includes utilities related to pseudo-random number
  generation.
* *core/ttest.py* includes utilities to compute a t-test test statistic on two
  samples.
* *core/variance.py* includes utilities to compute sample variance and pooled
  variance.
* *core/vector.py* provides an object, ie. `Vector`, which encapsulates a numpy
  array such that:
    * the number of dimensions of the array is known to be one,
    * the data type of the object is known to be float, and
    * the elements of the object are known to be finite.

### tests/

Includes functional tests and unit tests for part of the code in this package.
In practice, all tests would have been implemented but for time considerations
they were not all implemented.

Specifically, the file *tests/func/test_func_core_core.py* includes an
equivalent test to the one performed in the R code to validate the object
performing the power calculation,
ie. `UnpairedOneSidedPermutationTestPowerSimulator`.

## Usage

The code was tested using Python 3.8. It was assumed that the technical
constraint Python 3.6+ meant to use any version of Python above or equal to
Python 3.6. It was **not** assumed that it meant that the code should support
all versions of Python above or equal to Python 3.6.

Dependencies are given in the *requirements.txt*.

Here's an example showing how to perform the power calculation.

```angular2html
from core import UnpairedOneSidedPermutationTestPowerSimulator

simulator = UnpairedOneSidedPermutationTestPowerSimulator.make(seed=1234)
power = simulator.simulate(
number_of_simulations=300,
number_of_permutations=300,
number_of_observations=50,
means=(0.5, 0.),
scale=1.,
alpha=0.025
)
print(power)
```

## Performance

A thorough analysis of the code to improve performance was **not** performed due
to time limitation. Although, a dependency on the package `numba` was introduced
to improve performance. In practice, further analysis would be required to
improve the performance if the current performance of the code is not
sufficient.

Caching is activated for `numba`, thus the automated tests will run slower on
the first execution, and should run faster after the first execution.

## Documentation

A complete documentation of the code was **not** performed due to time
limitation. In practice, all the code would have been documented. Some comments
are available throughout the code.
