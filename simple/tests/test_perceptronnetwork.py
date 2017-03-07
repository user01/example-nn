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

    # def test_weight_update(self):
    #     """Test updating of weights"""
    #     network = PerceptronNetwork(
    #         [
    #             PerceptronLayer.blank(4, 4, 'layer1', ['a', 'b', 'c', 'd']),
    #             PerceptronLayer.blank(4, 2, 'layer2', ['a', 'b', 'c', 'd']),
    #             PerceptronLayer.blank(2, 2, 'layer3', ['a', 'b', 'c', 'd'])
    #         ]
    #     )
    #     _, layer_states = network.forward([1, 2, 1, 2])
    #     truths = [0.01, 0.15]
    #     learing_rate = 0.85

    #     unit_errors = network.backward(layer_states, truths)

    #     network_new = network.update_weights(
    #         layer_states, unit_errors, learing_rate)
    #     self.assertIsNotNone(network_new)
    #     self.assertNotEqual(network_new, network)
    #     self.assertEqual(network_new.shape(), network.shape())

    #     for layer_old, layer_new in zip(network.layers(), network_new.layers()):
    #         print('Layer {}'.format(layer_new.name()))
    #         for perceptron_old, perceptron_new in zip(layer_old.perceptrons(),
    #                                                   layer_new.perceptrons()):
    #             print('Perceptron {}'.format(perceptron_old.name()))
    #             self.assertNotEqual(perceptron_old.weights(),
    #                                 perceptron_new.weights())
    #             # for weight_old, weight_new in zip(perceptron_old.weights(),
    #             #                                   perceptron_new.weights()):
    #             #     self.assertNotEqual(weight_old, weight_new)

    def test_and(self):
        """Test network matches known AND weights"""
        epochs_to_run = 1
        learning_rate = 0.15
        value_inputs = [
            # A  B
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
        values_outputs = [1 if a + b == 2 else 0 for a, b in value_inputs]
        perceptron_simple = Perceptron([0.5, 0.5, 0.5], 'c', ['a', 'b'])

        for _ in range(0, epochs_to_run):
            # print('Starting Epoch {}'.format(epoch))
            standard_error = []
            for value, result in zip(value_inputs, values_outputs):
                # print('For {0}, truth {1} ...'.format(value, result))

                # Step 1: forward pass - predict
                estimated_value = perceptron_simple.forward(value)

                # Step 2: back pass - collect errors
                weighted_error = result - estimated_value
                standard_error.append(weighted_error ** 2)
                unit_error = perceptron_simple.backward(
                    estimated_value, weighted_error)

                # Step 3: update weights
                perceptron_simple = perceptron_simple.update_weights(
                    value, unit_error, learning_rate)

        network = PerceptronNetwork(
            [
                PerceptronLayer(
                    [
                        Perceptron([0.5, 0.5, 0.5], 'c', ['a', 'b'])
                    ], 'only_layer')
            ])
        values_network_outputs = [[a] for a in values_outputs]
        new_network, _, _, _ = network.train(
            value_inputs, values_network_outputs, learning_rate, epochs_to_run, 1000)

        self.assertEqual(1, len(new_network.layers()))

        layer = new_network.layers()[0]
        self.assertEqual(2, layer.input_size())
        self.assertEqual(1, layer.output_size())
        self.assertEqual(1, len(layer.perceptrons()))

        perceptron_layer = layer.perceptrons()[0]

        self.assertEqual(perceptron_simple.input_size(),
                         perceptron_layer.input_size())
        self.assertEqual(len(perceptron_simple.weights()),
                         len(perceptron_layer.weights()))
        for weight_simple, weight_layer, idx in zip(perceptron_simple.weights(),
                                                    perceptron_layer.weights(),
                                                    range(0, len(perceptron_layer.weights()))):
            self.assertAlmostEqual(
                weight_simple, weight_layer, 7, 'Weight mismatch index {}'.format(idx))


if __name__ == '__main__':
    unittest.main()
