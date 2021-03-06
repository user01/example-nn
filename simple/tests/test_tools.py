"""Test components for the simple nn"""

import unittest
import math
from simple.tools import linear_forward, sigmoid_forward
from simple.tools import linear_forward_verbose
from simple.tools import align_equations


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


class TestForwardVerbose(unittest.TestCase):
    """Forward Pass Verbosity tests"""

    def test_basic(self):
        """Basic"""
        inputs = [1, 2]
        weights = [3, 4, 5]
        name = "c"
        input_names = ['a', 'b']

        results_actual = linear_forward_verbose(
            inputs, weights, name, input_names)
        results_expected = [
            'net_c = w_c.BIAS * x_c.BIAS + w_c.a * x_c.a + w_c.b * x_c.b',
            '   17 =        3 *        1 +     4 *     1 +     5 *     2']
        self.assertListEqual(results_expected, results_actual)


class TestSigmoid(unittest.TestCase):
    """Sigmoid tests"""

    def test_basic(self):
        """Basic Sigmoid"""
        self.assertAlmostEqual(1 / (1 + math.exp(-5)), sigmoid_forward(5))

    def test_many(self):
        """Many Sigmoid"""
        for i in range(-100, 100):
            self.assertAlmostEqual(
                1 / (1 + math.exp(-i / 10)), sigmoid_forward(i / 10))


class TestAlignEquations(unittest.TestCase):
    """AlignEquations tests"""

    def test_basic(self):
        """Basic AlignEquations"""
        equations = [
            ['1', 'two', 'three'],
            ['four', 'two', '=']
        ]
        expected = [
            '   1 two three',
            'four two     ='
        ]
        actual = align_equations(equations)
        self.assertListEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
