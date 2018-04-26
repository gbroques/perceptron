import unittest

import numpy as np

from perceptron import Perceptron


class PerceptronTest(unittest.TestCase):

    @staticmethod
    def get_OR_training_data():
        """Data for learning the logical OR function.

        Returns:
            Data for learning logical OR.
        """
        return np.array([
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ])

    @staticmethod
    def get_AND_training_data():
        """Data for learning the logical AND function with sign function labels.

        Returns:
            Data for learning logical AND.
        """
        return np.array([
            [0, 0, -1],
            [0, 1, -1],
            [1, 0, -1],
            [1, 1, 1]
        ])

    def test_perceptron_with_heaviside(self):
        design_matrix, target_values = self.get_design_matrix_and_target_values('OR')
        perceptron = Perceptron(max_iter=100, learning_rate=0.2, activation_function='heaviside', seed=0)
        perceptron.fit(design_matrix, target_values)
        predictions = perceptron.predict(design_matrix)
        np.testing.assert_equal(predictions, target_values)

    def test_perceptron_with_sign(self):
        design_matrix, target_values = self.get_design_matrix_and_target_values('AND')
        perceptron = Perceptron(max_iter=100, learning_rate=0.2, activation_function='sign', seed=0)
        perceptron.fit(design_matrix, target_values)
        predictions = perceptron.predict(design_matrix)
        np.testing.assert_equal(predictions, target_values)

    def get_design_matrix_and_target_values(self, data: str):
        training_data = self.get_AND_training_data() if data == 'AND' else self.get_OR_training_data()
        design_matrix = training_data[:, :2]
        target_values = training_data[:, -1]
        return design_matrix, target_values

    def test_raises_value_error_for_invalid_activation_function(self):
        with self.assertRaises(ValueError):
            Perceptron(activation_function='bologna')

    def test_score(self):
        expected_score = 1.0
        design_matrix, target_values = self.get_design_matrix_and_target_values('OR')
        perceptron = Perceptron(max_iter=100, learning_rate=0.2, activation_function='heaviside', seed=0)
        perceptron.fit(design_matrix, target_values)
        score = perceptron.score(design_matrix, target_values)
        self.assertEqual(expected_score, score)


if __name__ == '__main__':
    unittest.main()
