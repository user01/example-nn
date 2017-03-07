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

    def test_update_weights_d(self):
        """Test if Perceptron D with updated weights are correct"""
        learning_rate = 0.3
        perceptron_d_input = 0.55  # checked via perceptron_c_output above
        perceptron_d_error = 0.11  # checked via perceptron_d_error above
        perceptron_d_new = self.perceptron_d.update_weights(
            [perceptron_d_input], perceptron_d_error, learning_rate)

        results_actual = perceptron_d_new.weights()
        #                   w_bias w_dc
        results_expected = [0.133, 0.11815]
        for actual, expected in zip(results_actual, results_expected):
            self.assertAlmostEqual(actual, expected)

    def test_update_weights_c(self):
        """Test if Perceptron C with updated weights are correct"""
        learning_rate = 0.3
        perceptron_c_input = VALUE_INPUTS[0]
        perceptron_c_error = 0.0028  # checked via perceptron_c_error above
        perceptron_c_new = self.perceptron_c.update_weights(
            perceptron_c_input, perceptron_c_error, learning_rate)

        results_actual = perceptron_c_new.weights()
        results_expected = [0.10084, 0.10084, 0.1]
        for actual, expected in zip(results_actual, results_expected):
            self.assertAlmostEqual(actual, expected)


class TestPerceptronAnd(unittest.TestCase):
    """Perceptron tests"""

    def test_simple_and(self):
        """Run an example with an AND logic"""

        value_inputs = [
            # A  B
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
        value_outputs = [0, 0, 0, 1]
        perceptron = Perceptron([0.5, 0.5, 0.5], 'c', ['a', 'b'])
        mse_current = float("inf")
        learning_rate = 0.25

        for epoch in range(0, 5000):
            standard_error = []
            for value, result in zip(value_inputs, value_outputs):

                # Step 1: forward pass - predict
                estimated_value = perceptron.forward(value)

                # Step 2: back pass - collect errors
                weighted_error = result - estimated_value
                standard_error.append(weighted_error ** 2)
                unit_error = perceptron.backward(
                    estimated_value, weighted_error)

                # Step 3: update weights
                perceptron = perceptron.update_weights(
                    value, unit_error, learning_rate)

            if epoch % 100 == 0:
                mse_new = sum(standard_error) / len(standard_error)
                # the MSE should always fall over many epochs
                self.assertLess(mse_new, mse_current)

        for value, result in zip(VALUE_INPUTS, value_outputs):
            estimated_value = perceptron.forward(value)
            self.assertLess(abs(result - estimated_value), 0.05)


if __name__ == '__main__':
    unittest.main()
