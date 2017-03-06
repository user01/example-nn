"""Test Perceptrons"""

import unittest

from simple.perceptron import Perceptron

VALUE_INPUTS = [
    # A  B
    [1, 0],
    [0, 1]
]
VALUE_OUTPUTS = [1, 0]


class TestPerceptron(unittest.TestCase):
    """Perceptron tests"""

    def setUp(self):
        self.perceptron_c = Perceptron([0.1, 0.1, 0.1], 'c', ['a', 'b'])
        self.perceptron_d = Perceptron([0.1, 0.1], 'd', ['c'])

    def test_forward(self):
        """Test the forward pass"""

        perceptron_c_output = self.perceptron_c.forward(VALUE_INPUTS[0])
        self.assertAlmostEqual(perceptron_c_output, 0.55, 2)

        perceptron_d_output = self.perceptron_d.forward([perceptron_c_output])
        self.assertAlmostEqual(perceptron_d_output, 0.54, 2)

    def test_backprop(self):
        """Test the backprop pass"""

        weighted_error_d = VALUE_OUTPUTS[0] - 0.54
        perceptron_d_error = self.perceptron_d.backward(0.54, weighted_error_d)
        self.assertAlmostEqual(perceptron_d_error, 0.11, 2)

        weighted_error_c = self.perceptron_d.weights()[1] * perceptron_d_error
        perceptron_c_error = self.perceptron_c.backward(0.54, weighted_error_c)
        self.assertAlmostEqual(perceptron_c_error, 0.0028, 4)


if __name__ == '__main__':
    unittest.main()
