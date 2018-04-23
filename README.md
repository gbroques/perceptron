# Perceptron
Perceptron learning algorithm implemented in Python.

![perceptron](perceptron.png)

## Getting Started
This project depends upon the popular numerical processing library [NumPy](http://www.numpy.org/) for lightning-fast vector arithmetic, and other packages for unit testing.

### Prerequisites

To install NumPy, it's recommended you use Python's offical package manager **pip**.

To ensure pip is installed on your machine, run the command:

```
$ pip --version
```

pip should come installed with Python depending upon your version.

For more details, see [installation](https://pip.pypa.io/en/stable/installing/)
on pip's documentation.

### Installing
It's recommended you use `virtualenv` to create isolated Python environments.

You can find details on [virtualenv's documentation](https://virtualenv.pypa.io/en/stable/).

Once pip is installed, run:

```
$ pip install -r requirements.txt
```

This will install this project's dependencies on your machine.

## How to Run

```
$ python main.py
```

## Usage
API inspired by the popular machine learning library
[scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html).

```python
import numpy as np
from perceptron import Perceptron

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
print(predictions)  # [0, 1, 1, 1]
```
