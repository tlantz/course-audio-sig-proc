import unittest
import A2Part1
import A2Part2
import numpy as np
import numpy.testing as npt

class Assignment2TestCase(unittest.TestCase):

    def test_genSine_should_generate_example_from_assignment(self):
        expected = np.array([0.54030231,
                             -0.63332387,
                             -0.93171798,
                             0.05749049,
                             0.96724906])
        actual = A2Part1.genSine(1.0, 10.0, 1.0, 50.0, 0.1)
        npt.assert_array_almost_equal(actual, expected)

    def test_genComplexSine_should_generate_example_from_assignment(self):
        expected = np.array([1.0 + 0.j,
                             0.30901699 - 0.95105652j,
                             -0.80901699 - 0.58778525j,
                             -0.80901699 + 0.58778525j,
                             0.30901699 + 0.95105652j])
        actual = A2Part2.genComplexSine(1.0, 5.0)
        npt.assert_array_almost_equal(actual, expected)
