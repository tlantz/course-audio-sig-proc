import unittest
import A3Part1
import numpy as np
from numpy import testing as npt

class Assignment3TestCase(unittest.TestCase):

    def _assert_zero_except(self, array, *non_zeros):
        nzs = set(non_zeros)
        for i in xrange(0, len(array)):
            if i in nzs:
                self.assertNotEqual(array[i], 0)
            else:
                npt.assert_almost_equal(array[i], 0)

    def _generate_two_sine_signal(self, fs, f1, f2):
        positions = np.arange(0, fs)
        time = np.arange(-1.0, 1.0, 1.0 / fs)
        sine1 = np.cos(2 * np.pi * f1 * time)
        sine2 = np.cos(2 * np.pi * f2 * time)
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
