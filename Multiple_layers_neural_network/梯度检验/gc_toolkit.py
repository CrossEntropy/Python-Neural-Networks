import numpy as np


def parameters_dictionary_to_vector(parameters):
    count = 0
    keys =[]
    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        new_vector = np.reshape(parameters[key], (-1, 1))
        keys = keys + [key] * new_vector.shape[0]
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), 0)
        count += 1
    return theta, keys


def parameters_vector_to_dictionary(theta):
    parameters ={}
    parameters["W1"] = np.reshape(theta[:40, :], (20, 2))
    parameters["b1"] = np.reshape(theta[40:60, :], (20, 1))
    parameters["W2"] = np.reshape(theta[60:120, :], (3, 20))
    parameters["b2"] = np.reshape(theta[120:123, :],(3, 1))
    parameters["W3"] = np.reshape(theta[123:126, :], (1, 3))
    parameters["b3"] = np.reshape(theta[126:127, :], (1, 1))
    return parameters


def gradient_dictionary_to_vector(grads):
    count = 0
    for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]:
        new_vector = np.reshape(grads[key], (-1, 1))
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), 0)
        count += 1
    return theta

