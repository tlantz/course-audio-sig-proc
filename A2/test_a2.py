import unittest
import A2Part1
import A2Part2
import A2Part3
import A2Part4
import A2Part5
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

    def test_DFT_should_generate_example_from_assignment(self):
        expected = np.array([10.0 + 0.0j,
                             -2. +2.0j,
                             -2.0 - 9.79717439e-16j,
                             -2.0 - 2.0j])
        actual = A2Part3.DFT(np.arange(1, 5))
        npt.assert_array_almost_equal(actual, expected)

    def test_IDFT_should_generate_example_from_assignment(self):
        expected = np.array([1.00000000e+00 +0.00000000e+00j,
                            -4.59242550e-17 +5.55111512e-17j,
                            0.00000000e+00 +6.12323400e-17j,
                            8.22616137e-17 +8.32667268e-17j])
        input = np.array([1, 1, 1, 1])
        actual = A2Part4.IDFT(input)
        npt.assert_array_almost_equal(actual, expected)

    def test_genMagSpec_should_generate_example_from_assignment(self):
        expected = np.array([10.0, 2.82842712, 2.0, 2.82842712])
        input = np.arange(len(expected)) + 1.0
        actual = A2Part5.genMagSpec(input)
        npt.assert_array_almost_equal(actual, expected)
