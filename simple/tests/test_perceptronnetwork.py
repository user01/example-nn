"""Test Perceptrons"""

import unittest

from simple.perceptron import PerceptronLayer, PerceptronNetwork


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

if __name__ == '__main__':
    unittest.main()
