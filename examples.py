"""Hard coded NN tests"""

from simple.perceptron import Perceptron
from simple.perceptronlayer import PerceptronLayer
from simple.perceptronnetwork import PerceptronNetwork

VALUE_INPUTS = [
    # A  B
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
LEARNING_RATE = 0.25
EPOCHS_AND = 5000


def line_break():
    """Simple line breaks"""
    print('')
    print(' =============================== ')
    print('')


def try_and():
    """Run an example with an AND logic"""
    line_break()
    print(' Simple single Perceptron AND')
    values_outputs = [1 if a + b == 2 else 0 for a, b in VALUE_INPUTS]
    perceptron = Perceptron([0.5, 0.5, 0.5], 'c', ['a', 'b'])

    for epoch in range(0, EPOCHS_AND):
        # print('Starting Epoch {}'.format(epoch))
        standard_error = []
        for value, result in zip(VALUE_INPUTS, values_outputs):
            # print('For {0}, truth {1} ...'.format(value, result))

            # Step 1: forward pass - predict
            estimated_value = perceptron.forward(value)

            # Step 2: back pass - collect errors
            weighted_error = result - estimated_value
            standard_error.append(weighted_error ** 2)
            unit_error = perceptron.backward(estimated_value, weighted_error)

            # Step 3: update weights
            perceptron = perceptron.update_weights(
                value, unit_error, LEARNING_RATE)

        if epoch % 1000 == 0:
            print('For epoch {0}, MSE of {1}'.format(
                epoch, sum(standard_error) / len(standard_error)))

    print('Final MSE {0}'.format(sum(standard_error) / len(standard_error)))
    print(' {0:>6} | {1:>5} {2:<10}'.format('Value', 'Truth', 'Prediction'))
    for value, result in zip(VALUE_INPUTS, values_outputs):
        estimated_value = perceptron.forward(value)
        print(' {0:>6} | {1:>5} {2:<10}'.format(
            str(value), result, round(estimated_value, 3)))


def report_network(data, truths, network):
    """Print function for Network Performance"""
    print(' {0:>6} | {1:>5} {2:<10}'.format('Value', 'Truth', 'Prediction'))
    for value, result in zip(data, truths):
        estimated_value, _ = network.forward(value)
        # print(value, result, estimated_value)
        print(' {0:>6} | {1:>5} {2:<10}'.format(
            str(value), result[0], round(estimated_value[0], 3)))


def try_and_network():
    """Run an example with Networked AND logic"""
    line_break()
    print('Network with AND')
    network = PerceptronNetwork.shorthand([2, 1])
    values_outputs = [[1] if a + b == 2 else [0] for a, b in VALUE_INPUTS]

    new_network, mse, mse_first, mses = network.train(
        VALUE_INPUTS, values_outputs, LEARNING_RATE, EPOCHS_AND, 1000)
    for epoch, mse in mses:
        print('For epoch {0}, MSE of {1}'.format(epoch, mse))
    print('From MSE of {} to {}'.format(mse_first, mse))

    report_network(VALUE_INPUTS, values_outputs, new_network)
    return new_network


def try_nor():
    """Run an example with NOR logic"""
    line_break()
    print('Network with NOR')
    network = PerceptronNetwork.shorthand([2, 1], 'relu')
    values_outputs = [[1] if a + b == 0 else [0] for a, b in VALUE_INPUTS]

    new_network, mse, mse_first, mses = network.train(
        VALUE_INPUTS, values_outputs, LEARNING_RATE, 5000, 1000)
    print('From MSE of {} to {}'.format(mse_first, mse))

    for epoch, mse in mses:
        print('For epoch {0}, MSE of {1}'.format(epoch, mse))

    print(' {0:>6} | {1:>5} {2:<10}'.format('Value', 'Truth', 'Prediction'))
    for value, result in zip(VALUE_INPUTS, values_outputs):
        estimated_value, _ = new_network.forward(value)
        # print(value, result, estimated_value)
        print(' {0:>6} | {1:>5} {2:<10}'.format(
            str(value), result[0], round(estimated_value[0], 3)))

    return new_network


def try_multi():
    """Run an example with Multi logic"""
    line_break()
    print('Network with !A & (B | !C)')
    network = PerceptronNetwork.shorthand([3, 4, 5, 4, 3, 1], 'relu')
    inputs = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ]
    values_outputs = [[1] if a == 0 and (b == 1 or c == 0) else [
        0] for a, b, c in inputs]

    new_network, mse, mse_first, mses = network.train(
        inputs, values_outputs, LEARNING_RATE, 5000, 1000)
    print('From MSE of {} to {}'.format(mse_first, mse))

    for epoch, mse in mses:
        print('For epoch {0}, MSE of {1}'.format(epoch, mse))

    print(' {0:>6} | {1:>5} {2:<10}'.format('Value', 'Truth', 'Prediction'))
    for value, result in zip(inputs, values_outputs):
        estimated_value, _ = new_network.forward(value)
        # print(value, result, estimated_value)
        print(' {0:>6} | {1:>5} {2:<10}'.format(
            str(value), result[0], round(estimated_value[0], 3)))

    return new_network


def print_notes(notes):
    """Print the notes in a pleasing format"""
    for note in notes:
        print(note)


def try_xor():
    """Run an example with XOR logic"""
    line_break()
    print('Network with XOR')
    network = PerceptronNetwork.shorthand([2, 3, 1], 'relu')
    network = PerceptronNetwork(
        [
            PerceptronLayer([
                Perceptron([0.1] * 3, 'A', ['X', 'Y'], 'sigmoid'),
                Perceptron([0.1] * 3, 'B', ['X', 'Y'], 'sigmoid'),
                Perceptron([0.1] * 3, 'C', ['X', 'Y'], 'sigmoid')
            ], 'main'),
            PerceptronLayer([
                Perceptron([0.1] * 4, 'D', ['A', 'B', 'C'], 'sigmoid')
            ], 'final')
        ]
    )
    print(network.shape())
    value_inputs = VALUE_INPUTS
    values_outputs = [[1] if a + b == 1 else [0] for a, b in VALUE_INPUTS]

    new_network, mse, mse_first, mses = network.train(
        value_inputs, values_outputs, LEARNING_RATE, 5000, 1000)
    print('From MSE of {} to {}'.format(mse_first, mse))

    for epoch, mse in mses:
        print('For epoch {0}, MSE of {1}'.format(epoch, mse))

    print(' {0:>6} | {1:>5} {2:<10}'.format('Value', 'Truth', 'Prediction'))
    for value, result in zip(value_inputs, values_outputs):
        estimated_value, _ = new_network.forward(value)
        # print(value, result, estimated_value)
        print(' {0:>6} | {1:>5} {2:<10}'.format(
            str(value), result[0], round(estimated_value[0], 3)))

    return new_network


def try_xor_verbose():
    """Run an example with XOR logic"""
    line_break()
    print('Network with XOR Verbose')
    # network = PerceptronNetwork.shorthand([2, 3, 1], 'relu')
    network = PerceptronNetwork(
        [
            PerceptronLayer([
                Perceptron([0.1] * 3, 'A', ['X', 'Y'], 'sigmoid'),
                Perceptron([0.1] * 3, 'B', ['X', 'Y'], 'sigmoid'),
                Perceptron([0.1] * 3, 'C', ['X', 'Y'], 'sigmoid')
            ], 'main'),
            PerceptronLayer([
                Perceptron([0.1] * 4, 'D', ['A', 'B', 'C'], 'sigmoid')
            ], 'final')
        ]
    )
    values_outputs = [[1] if a + b == 1 else [0] for a, b in VALUE_INPUTS]

    for epoch in range(0, 3):
        print('Starting Epoch {}'.format(epoch))
        standard_error = []
        for value, results in zip(VALUE_INPUTS, values_outputs):
            print(' ================================================== ')
            print(value, '->', results)
            print(' ================================================== ')

            estimated_results, network, notes = network.step(
                value, results, LEARNING_RATE)
            print_notes(notes)

            # Collect errors
            weighted_errors = [result - estimated_result for result,
                               estimated_result in zip(results, estimated_results)]
            weighted_error = sum(weighted_errors) / len(weighted_errors)
            standard_error.append(weighted_error ** 2)

        if epoch % 1 == 0:
            print('For epoch {0}, MSE of {1}'.format(
                epoch, sum(standard_error) / len(standard_error)))


def try_class_example_verbose():
    """Run an example from class, verbosely"""
    line_break()
    print('Network with class example, verbose')
    network = PerceptronNetwork(
        [
            PerceptronLayer([
                Perceptron([0.1] * 3, 'C', ['A', 'B'], 'sigmoid')
            ], 'main'),
            PerceptronLayer([
                Perceptron([0.1] * 2, 'D', ['C'], 'sigmoid')
            ], 'final')
        ]
    )
    print(network.shape())
    value_inputs = [[1, 0], [0, 1]]
    values_outputs = [[1], [0]]

    for epoch in range(0, 3):
        print('Starting Epoch {}'.format(epoch))
        standard_error = []
        for value, results in zip(value_inputs, values_outputs):
            print(' ================================================== ')
            print(value, '->', results)
            print(' ================================================== ')

            estimated_results, network, notes = network.step(
                value, results, 0.3)
            print_notes(notes)

            # Collect errors
            weighted_errors = [result - estimated_result for result,
                               estimated_result in zip(results, estimated_results)]
            weighted_error = sum(weighted_errors) / len(weighted_errors)
            standard_error.append(weighted_error ** 2)

        if epoch % 1 == 0:
            print('For epoch {0}, MSE of {1}'.format(
                epoch, sum(standard_error) / len(standard_error)))


def try_class_example():
    """Run an example from class"""
    line_break()
    print('Network with class example')
    network = PerceptronNetwork(
        [
            PerceptronLayer([
                Perceptron([0.1] * 3, 'C', ['A', 'B'], 'sigmoid')
            ], 'main'),
            PerceptronLayer([
                Perceptron([0.1] * 2, 'D', ['C'], 'sigmoid')
            ], 'final')
        ]
    )
    value_inputs = [[1, 0], [0, 1]]
    values_outputs = [[1], [0]]

    new_network, mse, mse_first, mses = network.train(
        value_inputs, values_outputs, LEARNING_RATE, 5000, 1000)
    print('From MSE of {} to {}'.format(mse_first, mse))

    for epoch, mse in mses:
        print('For epoch {0}, MSE of {1}'.format(epoch, mse))

    print(' {0:>6} | {1:>5} {2:<10}'.format('Value', 'Truth', 'Prediction'))
    for value, result in zip(value_inputs, values_outputs):
        estimated_value, _ = new_network.forward(value)
        # print(value, result, estimated_value)
        print(' {0:>6} | {1:>5} {2:<10}'.format(
            str(value), result[0], round(estimated_value[0], 3)))

    return new_network


if __name__ == "__main__":
    # try_and()
    try_and_network()
    try_nor()
    try_xor()
    # try_xor_verbose()
    try_multi()
    # try_class_example_verbose()
    try_class_example()
