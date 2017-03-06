"""Test components for the simple nn"""

import unittest
import math
from simple.tools import linear_forward, sigmoid_forward


class TestForwardPass(unittest.TestCase):
    """Forward Pass tests"""

    def test_basic(self):
        """Basic 1*2 + 1*3 = 5"""
        self.assertEqual(5, linear_forward([1], [2, 3]))

    def test_advanced(self):
        """Formula 1*2 + 2*3 = 8"""
        self.assertEqual(8, linear_forward([2], [2, 3]))

    def test_longer(self):
        """Formula 1*1 + 1*2 + 2*3 + 5*4 = 8"""
        self.assertEqual(29, linear_forward([1, 2, 5], [1, 2, 3, 4]))


class TestSigmoid(unittest.TestCase):
    """Sigmoid tests"""

    def test_basic(self):
        """Basic Sigmoid"""
        self.assertAlmostEqual(1 / (1 + math.exp(5)), sigmoid_forward(5))

    def test_many(self):
        """Many Sigmoid"""
        for i in range(-100, 100):
            self.assertAlmostEqual(1 / (1 + math.exp(i / 10)), sigmoid_forward(i / 10))


if __name__ == '__main__':
    unittest.main()
