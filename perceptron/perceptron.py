import sys
from time import sleep

import numpy as np
from numpy import ndarray

HEAVISIDE = 'heaviside'
SIGN = 'sign'

ACTIVATION_FUNCTIONS = {HEAVISIDE, SIGN}


class Perceptron:

    def __init__(self, max_iter=1000, learning_rate=1, activation_function='heaviside', seed=None):
        self._check_activation_function(activation_function)
        self._max_iter = max_iter
        self._learning_rate = learning_rate
        self._activation_function = activation_function
        self._random = np.random.RandomState(seed)
        self.weights_ = None

    @staticmethod
    def _check_activation_function(activation_function: str) -> None:
        if activation_function not in ACTIVATION_FUNCTIONS:
            raise ValueError('Invalid activation function {}.'.format(activation_function))

    def fit(self, design_matrix: ndarray, target_values: ndarray) -> 'Perceptron':
        m, n = design_matrix.shape
        design_matrix = self._append_bias(design_matrix, m)
        self.weights_ = self._random.rand(n + 1)  # + 1 for bias
        num_correct = 0
        print('Error rate:')
        for i in range(1, self._max_iter + 1):
            example, expected = self._get_random_example(design_matrix, target_values)
            actual = g(example.dot(self.weights_), self._activation_function)
            error = expected - actual
            num_correct += int(expected == actual)
            error_rate = (num_correct / i) * 100
            print_progress(round(error_rate))
            sleep(0.07)
            self.weights_ += self._learning_rate * error * example
        print('\n', end='')

        return self

    def predict(self, test_examples: ndarray) -> ndarray:
        m, n = test_examples.shape
        test_examples = self._append_bias(test_examples, m)
        a = np.vectorize(lambda x: g(x, self._activation_function))
        return a(np.dot(test_examples, self.weights_))

    @staticmethod
    def _append_bias(examples, m: int):
        bias = np.ones((m, 1))
        return np.append(examples, bias, axis=1)

    def _get_random_example(self, design_matrix, target_values):
        i = self._random.randint(0, target_values.size)
        example = design_matrix[i]
        target = target_values[i]
        return example, target

    def score(self, test_examples, target_values) -> float:
        predictions = self.predict(test_examples)
        num_correct = np.count_nonzero(predictions == target_values)
        return num_correct / (target_values.size * 1.0)


def g(input_value: float, activation_function='heaviside') -> int:
    """Activation function.

    See Also:
        https://en.wikipedia.org/wiki/Sign_function
        https://en.wikipedia.org/wiki/Heaviside_step_function

    Args:
        input_value: Input value.
        activation_function
    Returns:
        1 if the input is positive. -1 otherwise.
    """
    if activation_function == HEAVISIDE:
        return 1 if input_value > 0 else 0
    elif activation_function == SIGN:
        return 1 if input_value > 0 else -1


def print_progress(percent: int) -> None:
    bar = '\r[{0:<50}] {1}%'.format('=' * int(percent / 2), percent)
    sys.stdout.write(bar)
    sys.stdout.flush()
