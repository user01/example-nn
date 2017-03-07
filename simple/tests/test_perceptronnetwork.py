"""Test Perceptrons"""

import unittest

from simple.perceptron import Perceptron
from simple.perceptronlayer import PerceptronLayer
from simple.perceptronnetwork import PerceptronNetwork


class TestPerceptronNetwork(unittest.TestCase):
    """Perceptron Network tests"""

    def test_init(self):
        """Test the constructing a layer"""
        network = PerceptronNetwork(
            [
                PerceptronLayer.blank(4, 2, 'layer1', ['a', 'b', 'c', 'd']),
                PerceptronLayer.blank(2, 2, 'layer2', ['a', 'b', 'c', 'd'])
            ]
        )
        self.assertIsNotNone(network)

    def test_init_size_mismatch(self):
        """Test mismatching layer sizes"""
        with self.assertRaises(Exception):
            PerceptronNetwork(
                [
                    PerceptronLayer.blank(
                        4, 2, 'layer1', ['a', 'b', 'c', 'd']),
                    PerceptronLayer.blank(3, 2, 'layer2', ['a', 'b', 'c', 'd'])
                ]
            )

    def test_forward(self):
        """Test feeding forward"""
        network = PerceptronNetwork(
            [
                PerceptronLayer.blank(
                    4, 2, 'layer1', ['a', 'b', 'c', 'd']),
                PerceptronLayer.blank(2, 2, 'layer2', ['a', 'b', 'c', 'd'])
            ]
        )
        results_actual = network.forward([1, 2, 1, 2])[0]
        self.assertIsInstance(results_actual, list)
        self.assertEqual(2, len(results_actual))
        results_expected = [0.843, 0.843]

        for expected, actual in zip(results_expected, results_actual):
            self.assertAlmostEqual(actual, expected, 2)

    def test_backward(self):
        """Test feeding backward"""
        network = PerceptronNetwork(
            [
                PerceptronLayer.blank(4, 4, 'layer1', ['a', 'b', 'c', 'd']),
                PerceptronLayer.blank(4, 2, 'layer2', ['a', 'b', 'c', 'd']),
                PerceptronLayer.blank(2, 2, 'layer3', ['a', 'b', 'c', 'd'])
            ]
        )
        _, layer_states = network.forward([1, 2, 1, 2])
        truths = [1, 0.5]

        unit_errors_actual = network.backward(layer_states, truths)
        self.assertEqual(3, len(unit_errors_actual))
        self.assertEqual(4, len(unit_errors_actual[0]))
        self.assertEqual(2, len(unit_errors_actual[1]))
        self.assertEqual(2, len(unit_errors_actual[2]))

    def test_backward_2(self):
        """Test feeding backward"""
        network = PerceptronNetwork(
            [
                PerceptronLayer.blank(2, 3, 'input_layer', ['a', 'b']),
                PerceptronLayer.blank(
                    3, 3, 'hidden_layer', ['input_a', 'input_b', 'input_c']),
                PerceptronLayer.blank(
                    3, 1, 'output_layer', ['hidden_a', 'hidden_b', 'hidden_c'])
            ]
        )
        _, layer_states = network.forward([1, 0.5])
        truths = [1]

        unit_errors_actual = network.backward(layer_states, truths)
        self.assertEqual(3, len(unit_errors_actual))
        self.assertEqual(3, len(unit_errors_actual[0]))
        self.assertEqual(3, len(unit_errors_actual[1]))
        self.assertEqual(1, len(unit_errors_actual[2]))

    def test_and_single_epoch(self):
        """Test network matches known AND weights with one epoch"""
        learning_rate = 0.15
        value_inputs = [
            # A  B
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
        values_simple_outputs = [1 if a + b ==
                                 2 else 0 for a, b in value_inputs]
        values_network_outputs = [[a] for a in values_simple_outputs]
        perceptron = Perceptron([0.5, 0.5, 0.5], 'c', ['a', 'b'])
        network = PerceptronNetwork(
            [
                PerceptronLayer(
                    [
                        Perceptron([0.5, 0.5, 0.5], 'c', ['a', 'b'])
                    ], 'only_layer')
            ])

        perceptron_estimated_values = []
        network_estimated_values = []
        perceptron_unit_error = []
        network_unit_error = []
        for value, result in zip(value_inputs, values_simple_outputs):
            # Step 1: forward pass - predict
            estimated_value = perceptron.forward(value)
            perceptron_estimated_values.append(estimated_value)

            # Step 2: back pass - collect errors
            weighted_error = result - estimated_value
            unit_error = perceptron.backward(
                estimated_value, weighted_error)
            perceptron_unit_error.append(unit_error)

            # Step 3: update weights
            perceptron_updated = perceptron.update_weights(
                value, unit_error, learning_rate)

        for values, results in zip(value_inputs, values_network_outputs):
            # Step 1: forward pass - predict
            estimated_results, layer_states = network.forward(values)
            network_estimated_values.append(estimated_results[0])

            # Step 2: back pass - collect errors
            unit_errors = network.backward(layer_states, results)
            network_unit_error.append(unit_errors[0][0])

            # Step 3: update weights
            network_updated = network.update_weights(
                layer_states, unit_errors, learning_rate)

        self.assertEqual(len(perceptron_estimated_values),
                         len(network_estimated_values))
        for result_perceptron, results_network in zip(perceptron_estimated_values,
                                                      network_estimated_values):
            self.assertAlmostEqual(result_perceptron, results_network)
        for unit_error_perceptron, unit_errors_network in zip(perceptron_unit_error,
                                                              network_unit_error):
            self.assertAlmostEqual(unit_error_perceptron, unit_errors_network)

        self.assertEqual(1, len(network_updated.layers()))

        layer = network_updated.layers()[0]
        self.assertEqual(2, layer.input_size())
        self.assertEqual(1, layer.output_size())
        self.assertEqual(1, len(layer.perceptrons()))

        perceptron_layer = layer.perceptrons()[0]

        self.assertEqual(perceptron_updated.input_size(),
                         perceptron_layer.input_size())
        self.assertEqual(len(perceptron_updated.weights()),
                         len(perceptron_layer.weights()))
        for weight_simple, weight_layer, idx in zip(perceptron_updated.weights(),
                                                    perceptron_layer.weights(),
                                                    range(0, len(perceptron_layer.weights()))):
            self.assertAlmostEqual(
                weight_simple, weight_layer, 7, 'Weight mismatch index {}'.format(idx))


if __name__ == '__main__':
    unittest.main()
