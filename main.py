import numpy as np

from perceptron import Perceptron


def main():
    # Training data for logical OR function
    training_data = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])
    design_matrix = training_data[:, :2]
    target_values = training_data[:, -1]

    perceptron = Perceptron(max_iter=100, learning_rate=0.2)
    perceptron.fit(design_matrix, target_values)
    predictions = perceptron.predict(design_matrix)
    print(predictions)


if __name__ == '__main__':
    main()
