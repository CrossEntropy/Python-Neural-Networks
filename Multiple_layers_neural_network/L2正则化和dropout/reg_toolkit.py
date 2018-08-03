import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from CLASS_2.week1.assignment_2_regularization.L2_and_drop_out.testCases import *


def relu(Z):
    A = np.maximum(0, Z)
    return A


def sigmoid(Z):
    A = 1. / (1 + np.exp(-Z))
    return A


def initialize_parameters_He(layers_size):
    np.random.seed(3)
    parameters = {}
    L = len(layers_size)
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_size[l], layers_size[l-1]) * np.sqrt(1/layers_size[l-1])
        parameters["b" + str(l)] = np.zeros(shape=(layers_size[l], 1))
    return parameters


# 直接构造三层神经网络
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = relu(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = sigmoid(Z3)
    cache = (X, W1, Z1, A1, W2, Z2, A2, W3, Z3)
    return A3, cache


# 构造dropout的前向传播
def forward_propagation_dropout(X, parameters, keep_prob):
    """
    :param X: 输入
    :param parameters: 参数
    :param keep_prob: 每个神经元存活的概率
    :return: A3, cache
    """
    np.random.seed(1)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    D1 = (D1 < keep_prob)
    A1 = A1 * D1
    A1 /= keep_prob  # 保证期望值

    Z2 = W2.dot(A1) + b2
    A2 = relu(Z2)
    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = (D2 < keep_prob)
    A2 = A2 * D2
    A2 /= keep_prob  # 保证期望值

    Z3 = W3.dot(A2) + b3
    A3 = sigmoid(Z3)  # 输出层输入层都不执行dropout
    caches = (X, W1, Z1, D1, A1, W2, Z2, D2, A2, W3, Z3)
    return A3, caches


def compute_cost(A3, Y):
    m = Y.shape[1]
    logprobs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = np.nansum(logprobs) / m
    return cost


def compute_cost_L2(A3, Y, paramters, lambd):
    cross_entropy_cost = compute_cost(A3, Y)
    m = Y.shape[1]
    sum_weight = 0
    L = len(paramters) // 2
    for l in range(1, L+1):
        W = paramters["W" + str(l)]
        sum_weight += np.sum(W**2)
    L2_regularization_cost = sum_weight * lambd / (2 * m)
    cost = cross_entropy_cost + L2_regularization_cost
    return cost


def backward_propagation(A3, Y, caches):
    grads = {}
    m = Y.shape[1]
    (X, W1, Z1, A1, W2, Z2, A2, W3, Z3) = caches
    dZ3 = A3 - Y
    dW3 = dZ3.dot(A2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m
    dA2 = W3.T.dot(dZ3)

    dZ2 = dA2 * np.where(Z2>0, 1, 0)
    dW2 = dZ2.dot(A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dA1 = W2.T.dot(dZ2)

    dZ1 = dA1 * np.where(Z1>0, 1, 0)
    dW1 = dZ1.dot(X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    grads["dW1"] = dW1
    grads["db1"] = db1
    grads["dW2"] = dW2
    grads["db2"] = db2
    grads["dW3"] = dW3
    grads["db3"] = db3
    return grads


def backward_propagation_L2(A3, Y, caches, parameters, lambd):
    grads = backward_propagation(A3, Y, caches)
    m = Y.shape[1]
    for l in range(1, 4):
        W = parameters["W" + str(l)]
        grads["dW" + str(l)] = grads["dW" + str(l)] + lambd * W / m
    return grads


def backward_propagation_dropout(A3, Y, caches, keep_prob):
    grads = {}
    m = Y.shape[1]
    (X, W1, Z1, D1, A1, W2, Z2, D2, A2, W3, Z3) = caches
    # (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = caches
    dZ3 = A3 - Y
    dW3 = dZ3.dot(A2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m
    dA2 = W3.T.dot(dZ3)
    dA2 *= D2
    dA2 /= keep_prob

    dZ2 = dA2 * np.where(A2 > 0, 1, 0)
    dW2 = dZ2.dot(A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dA1 = W2.T.dot(dZ2)
    dA1 *= D1
    dA1 /= keep_prob

    dZ1 = dA1 * np.where(A1 > 0, 1, 0)
    dW1 = dZ1.dot(X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads["dW1"] = dW1
    grads["db1"] = db1
    grads["dW2"] = dW2
    grads["db2"] = db2
    grads["dW3"] = dW3
    grads["db3"] = db3
    grads["dA1"] = dA1
    grads["dA2"] = dA2
    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L+1):
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        dW = grads["dW" + str(l)]
        db = grads["db" + str(l)]
        parameters["W" + str(l)] = W - learning_rate * dW
        parameters["b" + str(l)] = b - learning_rate * db
    return parameters


def predict_accuracy(X, Y, parameters):
    A3, caches = forward_propagation(X, parameters)
    prediction = np.where(A3 > 0.5, 1, 0)
    accuracy = np.mean(np.where(prediction==Y, 1, 0))
    return accuracy


def predict_dec(parameters, X):
    """
    Used for plotting decision boundary.

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (m, K)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Predict using forward propagation and a classification threshold of 0.5
    a3, cache = forward_propagation(X, parameters)
    predictions = (a3 > 0.5)
    return predictions


# 加载二维数据集
def load_2D_datasets():
    data = scipy.io.loadmat("E:\\Python-Neural-Network\\Multiple_layers_neural_network\\L2正则化和dropout\\data.mat")
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T
    return train_X, train_Y, test_X, test_Y


# 画决策边界
def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y[0, :], cmap=plt.cm.Spectral)
    plt.show()


if __name__ =="__main__":
    # 检验反向传播
    # X_assess, Y_assess, cache = backward_propagation_with_dropout_test_case()
    #
    # gradients = back_propagation_dropout(X_assess, Y_assess, cache, keep_prob=0.8)
    #
    # print("dA1 = " + str(gradients["dA1"]))
    # print("dA2 = " + str(gradients["dA2"]))
    # 检验前向传播
    X_assess, parameters = forward_propagation_with_dropout_test_case()

    A3, cache = forward_propagation_dropout(X_assess, parameters, keep_prob=0.7)
    print("A3 = " + str(A3))
