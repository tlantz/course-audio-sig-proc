import unittest
import A3Part1
import A3Part2
import A3Part3
import numpy as np
from numpy import testing as npt


class Assignment3TestCase(unittest.TestCase):

    def _assert_zero_except(self, array, *non_zeros):
        '''Checks for logical "zero" (negative) because this is decibals.'''
        nzs = set(non_zeros)
        for i in xrange(0, len(array)):
            if i in nzs:
                self.assertGreater(array[i], -120.0)
            else:
                self.assertLess(array[i], -120.0)

    def _generate_sine(self, fs, f):
        time = np.arange(-1.0, 1.0, 1.0 / fs)
        sine = np.cos(2 * np.pi * f * time)
        return sine

    def _generate_two_sine_signal(self, fs, f1, f2):
        sine1 = self._generate_sine(fs, f1)
        sine2 = self._generate_sine(fs, f2)
        return sine1 + sine2

    def test_minimizeEnergySpreadDFT_has_expected_bins_1(self):
        fs = 10000
        f1 = 80
        f2 = 200
        x = self._generate_two_sine_signal(fs, f1, f2)
        bins = A3Part1.minimizeEnergySpreadDFT(x, fs, f1, f2)
        self.assertEquals(126, len(bins))
        self._assert_zero_except(bins, 2, 5)

    def test_minimizeEnergySpreadDFT_has_expected_bins_2(self):
        fs = 48000
        f1 = 300
        f2 = 800
        x = self._generate_two_sine_signal(fs, f1, f2)
        bins = A3Part1.minimizeEnergySpreadDFT(x, fs, f1, f2)
        self.assertEquals(241, len(bins))
        self._assert_zero_except(bins, 3, 8)

    def _test_optimalZeropad(self, length, maxbin_index, fs, f, M):
        x = self._generate_sine(fs, f)[:M]
        mX = A3Part2.optimalZeropad(x, fs, f)
        self.assertEquals(length, len(mX))
        maxbin = mX[maxbin_index]
        self.assertGreater(maxbin, mX[maxbin_index - 1])
        self.assertGreater(maxbin, mX[maxbin_index + 1])

    def test_optimalZeropad_1(self):
        self._test_optimalZeropad(16, 3, 1000, 100, 25)

    def test_optimalZeropad_2(self):
        self._test_optimalZeropad(121, 6, 10000, 250, 210)

    def test_testRealEven_1(self):
        x = np.array([2, 3, 4, 3, 2])
        is_real_even, dftbuffer, X = A3Part3.testRealEven(x)
        self.assertTrue(is_real_even)
        npt.assert_array_equal(
            dftbuffer,
            np.array([4., 3., 2., 2., 3.]))
        npt.assert_array_almost_equal(
            X,
            np.array([14.0000 + 0.j,
                      2.6180 + 0.j,
                      0.3820 + 0.j,
                      0.3820 + 0.j,
                      2.6180 + 0.j]),
            4)

    def test_testRealEven_2(self):
        x = np.array([1, 2, 3, 4, 1, 2, 3])
        is_real_even, dftbuffer, X = A3Part3.testRealEven(x)
        self.assertFalse(is_real_even)
        npt.assert_array_equal(
            dftbuffer,
            np.array([4., 1., 2., 3., 1., 2., 3.]))
        npt.assert_array_almost_equal(
            X,
            np.array([16. + 0.j,
                      2. + 0.69j,
                      2. + 3.51j,
                      2. - 1.08j,
                      2. + 1.08j,
                      2. - 3.51j,
                      2. - 0.69j]),
            2)
