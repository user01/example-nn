"""Test Perceptrons"""

import unittest

from simple.perceptron import Perceptron

class TestPerceptron(unittest.TestCase):
    """Perceptron tests"""

    def test_basic(self):
        """Basic Perceptron"""

        value_inputs = [
            # A  B
            [1, 0],
            [0, 1]
        ]
        # learning_rate = 0.3

        perceptron_c = Perceptron([0.1, 0.1, 0.1], 'c', ['a', 'b'])

        # perceptron_c_output = perceptron_c.forward(value_inputs[0])
        # self.assertAlmostEqual(perceptron_c_output, 0.55)


if __name__ == '__main__':
    unittest.main()
