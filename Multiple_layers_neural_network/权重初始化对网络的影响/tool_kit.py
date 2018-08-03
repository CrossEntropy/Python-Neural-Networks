import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A


def relu(Z):
    A = np.maximum(0, Z)
    return A


def sigmoid_backward(dA, Z):
    A = sigmoid(Z)
    dZ = dA * A * (1-A)
    return dZ


def relu_backward(dA, Z):
    s = np.where(Z>0, 1, 0)
    dZ = dA * s
    return dZ


def initialize_parameters_random(layers_size):
    np.random.seed(3)
    parameters = {}
    L = len(layers_size)
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_size[l], layers_size[l-1]) *10
        parameters["b" + str(l)] = np.zeros(shape=(layers_size[l], 1))
    return parameters


def initialize_parameters_He(layers_size):
    np.random.seed(3)
    parameters = {}
    L = len(layers_size)
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_size[l], layers_size[l - 1]) * np.sqrt(2/layers_size[l-1])
        parameters["b" + str(l)] = np.zeros(shape=(layers_size[l], 1))
    return parameters


def initialize_parameters_zeros(layers_size):
    parameters = {}
    L = len(layers_size)
    for l in range(1, L):
        parameters["W" + str(l)] = np.zeros(shape=(layers_size[l], layers_size[l - 1]))
        parameters["b" + str(l)] = np.zeros(shape=(layers_size[l], 1))
    return parameters


def linear_forward(A_prev, W, b):
    Z = W.dot(A_prev) + b
    cache_linear = (A_prev, W)
    return Z, cache_linear


def activation_forward(Z, activation):
    if activation=="relu":
        A = relu(Z)
    elif activation=="sigmoid":
        A = sigmoid(Z)
    cache_activation = Z
    return A, cache_activation


def linear_activation_forward(A_prev, W, b, activation):
    Z, cache_linear = linear_forward(A_prev, W, b)
    A, cache_activation = activation_forward(Z, activation)
    cache = (cache_linear, cache_activation)
    return A, cache


def forward_propagation(X, parameters):
    L = len(parameters)//2
    A = X
    caches = []
    for l in range(1, L):
        A_prev = A
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        A, cache = linear_activation_forward(A_prev, W, b, "relu")
        caches.append(cache)
    W = parameters["W" + str(L)]
    b = parameters["b" + str(L)]
    AL, cache = linear_activation_forward(A, W, b, "sigmoid")
    caches.append(cache)
    return AL, caches


def linear_backward(dZ, cache_linear):
    A_prev, W = cache_linear
    m = dZ.shape[1]
    dW = dZ.dot(A_prev.T) /m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = W.T.dot(dZ)
    return dA_prev, dW, db


def activation_backward(dA, cache_activation, activation):
    Z = cache_activation
    if activation=="relu":
        dZ = relu_backward(dA, Z)
    elif activation=="sigmoid":
        dZ = sigmoid_backward(dA, Z)
    return dZ


def linear_activation_backward(dA, cache, activation):
    cache_linear, cache_activation = cache
    dZ = activation_backward(dA, cache_activation, activation)
    dA_prev, dW, db = linear_backward(dZ, cache_linear)
    return dA_prev, dW, db


def backward_propagation(AL, Y, caches):
    grads ={}
    L = len(caches)
    dZ = AL - Y
    cache_linear, cache_activation = caches[-1]
    dA_prev, dW, db = linear_backward(dZ, cache_linear)
    grads["dW" +str(L)] = dW
    grads["db" + str(L)] = db
    grads["dA" + str(L-1)] = dA_prev
    for l in reversed(range(1, L)):
        dA = grads["dA" + str(l)]
        cache = caches[l-1]
        dA_prev, dW, db = linear_activation_backward(dA_prev, cache, "relu")
        grads["dW" + str(l)] = dW
        grads["db" + str(l)] = db
        grads["dA" + str(l-1)] = dA_prev
    return grads


def compute_cost(AL, Y):
    logprobs = -Y*np.log(AL) - (1-Y) * np.log(1-AL)
    m = Y.shape[1]
    cost = np.nansum(logprobs) / m
    return cost


def update_parameters(parameters, grads, learning_rate=0.01):
    L = len(parameters) // 2
    for l in range(1, L+1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate *grads["dW" +str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
    return parameters


def predict(X, Y, parameters):
    AL, caches = forward_propagation(X, parameters)
    prediction = np.where(AL>0.5, 1, 0)
    accuracy = np.mean(np.where(prediction==Y, 1, 0))
    return accuracy


def model(X, Y, layers_size, initialization, learning_rate=0.01, num_iterations=15000, print_cost=True):
    if initialization == "random":
        parameters = initialize_parameters_random(layers_size)
    elif initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_size)
    elif initialization == "He":
        parameters = initialize_parameters_He(layers_size)
    costs = []
    for i in range(num_iterations):
        AL, caches = forward_propagation(X, parameters)
        cost = compute_cost(AL, Y)
        if i%1000==0 and print_cost:
            print("第{}步的cost是{}".format(i, cost))
            costs.append(cost)
        grads = backward_propagation(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
    plt.plot(costs)
    plt.title("learning_rate is "+str(learning_rate))
    plt.xlabel("iteration (per thousands)")
    plt.ylabel("costs")
    plt.show()
    return parameters


def load_dataset():
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y


if __name__ == "__main__":
    train_X, train_Y, test_X, test_Y = load_dataset()
    plt.show()
    print("初始化方法选择 Zeros:")
    parameters = model(train_X, train_Y, [2, 10, 5, 1], initialization="zeros")
    print("on the train set:")
    print(predict(train_X, train_Y, parameters))
    print("on the test set:")
    print(predict(test_X, test_Y, parameters))
    print("初始化方法选择 random:")
    parameters = model(train_X, train_Y, [2, 10, 5, 1], initialization="random")
    print("on the train set:")
    print(predict(train_X, train_Y, parameters))
    print("on the test set:")
    print(predict(test_X, test_Y, parameters))
    print("初始化方法选择 He:")
    parameters = model(train_X, train_Y, [2, 10, 5, 1], initialization="He")
    print("on the train set:")
    print(predict(train_X, train_Y, parameters))
    print("on the test set:")
    print(predict(test_X, test_Y, parameters))