import unittest
import A4Part1

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
