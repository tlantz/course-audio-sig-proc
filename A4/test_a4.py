import unittest
import A4Part1
import A4Part2
import sys
import numpy.testing as npt

class Assignment4Testcase(unittest.TestCase):

    def test_extractMainLobe_case_1(self):
        # Blackman Harris 100 should be of size 65
        mX = A4Part1.extractMainLobe('blackmanharris', 100)
        self.assertEqual(len(mX), 65)

    def test_extractMainLobe_case_2(self):
        # Boxcar 120 should be of size 17
        mX = A4Part1.extractMainLobe('boxcar', 120)
        self.assertEqual(len(mX), 17)

    def test_extractMainLobe_case_3(self):
        # Hamming 256 should be of size 33
        mX = A4Part1.extractMainLobe('hamming', 256)
        self.assertEqual(len(mX), 33)

    def test_computerSNR_case_1(self):
        snr, snr_part = A4Part2.computeSNR(
            '../../sounds/piano.wav',
            'blackman',
            513,
            2048,
            128)
        npt.assert_almost_equal(snr, 67.57, decimal=0)
        npt.assert_almost_equal(snr_part, 304.68, decimal=0)

    def test_computerSNR_case_2(self):
        snr, snr_part = A4Part2.computeSNR(
            '../../sounds/sax-phrase-short.wav',
            'hamming',
            512,
            1024,
            64)
        npt.assert_almost_equal(snr, 89.51, decimal=0)
        npt.assert_almost_equal(snr_part, 306.19, decimal=0)

    def test_computerSNR_case_3(self):
        snr, snr_part = A4Part2.computeSNR(
            '../../sounds/rain.wav',
            'hann',
            1024,
            2048,
            128)
        npt.assert_almost_equal(snr, 74.63, decimal=0)
        self.assertLess(abs(snr_part - 304.27), 10.)
