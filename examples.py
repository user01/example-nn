"""Hard coded NN tests"""

from simple.perceptron import Perceptron
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


def try_and():
    """Run an example with an AND logic"""
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
    """Run an example with XOR logic"""
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

def try_xor():
    """Run an example with XOR logic"""
    network = PerceptronNetwork.shorthand([2, 3, 1], 'relu')
    values_outputs = [[1] if a + b == 1 else [0] for a, b in VALUE_INPUTS]

    new_network, mse, mse_first, mses = network.train(
        VALUE_INPUTS, values_outputs, LEARNING_RATE, 10000, 2000)
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

if __name__ == "__main__":
    # try_and()
    # try_and_network()
    try_nor()
    try_xor()
