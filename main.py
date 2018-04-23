from argparse import ArgumentParser

import numpy as np
from numpy import ndarray

from perceptron import Perceptron


def main():
    data = read_data()
    m, n = data.shape
    num_features = n - 1
    design_matrix = data[:, :num_features]
    target_values = data[:, -1]

    perceptron = Perceptron(max_iter=100, learning_rate=0.2)
    perceptron.fit(design_matrix, target_values)

    wait_for_test_example(perceptron, num_features)


def read_data() -> ndarray:
    """Read in data from a text file."""
    filename = get_filename()
    with open(filename) as f:
        f.readline()  # Skip first line
        data = []
        for line in f:
            row = line.split()
            data.append([int(num) for num in row])
    return np.array(data)


def get_filename() -> str:
    """Get the filename to read from as the first command line argument."""
    parser = ArgumentParser(description='Train a perceptron with data stored in a text file.')
    parser.add_argument('filename',
                        metavar='filename',
                        type=str,
                        help='Data should be in a specific format. See data.txt for an example.')
    args = parser.parse_args()
    filename = vars(args)['filename']
    return filename


def wait_for_test_example(perceptron: Perceptron, num_features: int) -> None:
    print('Waiting for test examples. Press Ctrl+C to quit.')
    while True:
        try:
            line = input('Please enter a test example delimited by whitespace: ')
            row = line.split()
            example = np.array([[int(num) for num in row]])
            if len(example[0]) != num_features:
                raise ValueError()
            prediction = perceptron.predict(example)
            print('The model predicts {}'.format(prediction[0]))
        except ValueError as e:
            print(e)


if __name__ == '__main__':
    main()
