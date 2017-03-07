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
        """Test feeding forward"""
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


if __name__ == '__main__':
    unittest.main()
