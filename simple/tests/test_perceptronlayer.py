"""Test Perceptrons"""

import unittest

from simple.perceptron import Perceptron
from simple.perceptronlayer import PerceptronLayer


class TestPerceptronLayer(unittest.TestCase):
    """Perceptron Layer tests"""

    def test_init(self):
        """Test the constructing a layer"""
        new_layer = PerceptronLayer(
            [
                Perceptron([1, 2], 'a', ['i']),
                Perceptron([1, 3], 'b', ['i'])
            ],
            'layer1'
        )
        self.assertEqual(2, len(new_layer.perceptrons()))
        self.assertEqual(1, new_layer.perceptrons()[0].input_size())

    def test_init_blank(self):
        """Test the constructing a blank layer"""
        new_layer = PerceptronLayer.blank(4, 2, 'layer1', ['a', 'b', 'c', 'd'])
        self.assertEqual(2, len(new_layer.perceptrons()))
        self.assertEqual(4, new_layer.perceptrons()[0].input_size())

    def test_forward(self):
        """Test feeding forward through a layer"""
        # new_layer = PerceptronLayer.blank(4, 2, 'layer1', ['a', 'b', 'c', 'd'])
        new_layer = PerceptronLayer([
            Perceptron([0.1, 0.2, 0.3, 0.4, 0.5], 'C', ['a', 'b', 'c', 'd'], 'sigmoid')
        ], 'main')
        results_actual = new_layer.forward([1, 2, 1, 2])
        results_expected = [0.9088, 0.958]
        for expected, actual in zip(results_expected, results_actual):
            self.assertAlmostEqual(actual, expected, 2)

    # def test_backward(self):
    #     """Test feeding backward through a layer"""
    #     new_layer = PerceptronLayer.blank(4, 2, 'layer1', ['a', 'b', 'c', 'd'])
    #     example_outputs = [0.958, 0.958]
    #     example_errors = [0.05, 0.05]
    #     unit_errors_actual = new_layer.backward(
    #         example_outputs, example_errors)
    #     unit_errors_expected = [0.002, 0.002]

    #     for expected, actual in zip(unit_errors_expected, unit_errors_actual):
    #         self.assertAlmostEqual(actual, expected, 2)

    def test_update_weights(self):
        """Test update_weights via a layer"""
        layer = PerceptronLayer.blank(4, 2, 'layer1', ['a', 'b', 'c', 'd'])

        example_inputs = [0.958, 0.958]
        example_errors = [0.05, 0.05]
        layer_updated = layer.update_weights(
            example_inputs, example_errors, 0.05)

        perceptrons_old = layer.perceptrons()
        perceptrons_new = layer_updated.perceptrons()

        self.assertEqual(len(perceptrons_new), len(perceptrons_old))

        for perceptron_old, perceptron_new in zip(perceptrons_old, perceptrons_new):
            for weight_old, weight_new in zip(perceptron_old.weights(), perceptron_new.weights()):
                self.assertNotEqual(weight_old, weight_new)


if __name__ == '__main__':
    unittest.main()
