import numpy as np
import h5py

"""
step1: 初始化参数
step2: loop: 
        1. 前向传播
        2. 计算cost
        3. 反向传播
        4. 更新参数
step3:  返回参数，就可以构造预测函数进行预测了     
"""


# 参数初始化
def initialize_parameters(dims_layer):
    """
    参数初始化
    :param dims_layer: 每层的节点数， 从0层到L层。 其中0层是输入层
    :return: 返回一个python字典，存储着"W1","b1","W2", "b2"
    """
    L = len(dims_layer) - 1
    parameters = {}
    for l in range(1, L+1):
        parameters["W" + str(l)] = np.random.randn(dims_layer[l], dims_layer[l-1]) / dims_layer[l-1]
        parameters["b" + str(l)] = np.zeros(shape=(dims_layer[l], 1))
    return parameters

# 前向传播：  1. single layer linear, cache_linear=(A_prev, W, b)
#           2. single layer activation, cache_activation= Z
#           3. single layer linear --> activation,合并 cache_forward=(cache_linear, cache_activation)
#           4. multilayer : layer_0-->layer_1-------->layer_L, caches=(cache_forward_1,......cache_forward_L)


def linear_forward(A_prev, W, b):
    """
    在一层上实现 Z = W*X +b
    :param A_prev: 前一层的激活函数输出矩阵，shape=(n_l-1, m)
    :param W: 当前层的权重矩阵，shape=(n_L, n_l-1)
    :param b: 当前层的偏置矩阵，shape=(n_l, 1)
    :return: 1.返回加权和（线性组合）Z，shape=(n_l, m)
             2. 返回cache_linear, 里面缓存着 A_prev, W, b，方便于反向传播计算
    """
    assert A_prev.shape[0] == W.shape[1]
    Z = W.dot(A_prev) + b
    cache_linear = (A_prev, W, b)
    return Z, cache_linear


def sigmoid_forward(Z):
    """
    sigmoid激活函数输出
    :param Z: 加权和 Z
    :return: 1. sigmoid激活函数输出A
             2. cache_activation = Z，方便反向传播计算
    """
    A = 1 / (1 + np.exp(-Z))
    cache_activation = Z
    return A, cache_activation


def relu_forward(Z):
    """
    relu激活函数输出
    :param Z: 加权和Z
    :return: 1. relu激活函数输出A
             2. cache_activation = Z, 方便反向传播计算
    """
    A = np.maximum(Z, 0)
    cache_activation = Z
    return A, cache_activation


def linear_activation_forward(A_prev, W, b, activation):
    """
    将linear,activation整合在一起
    :param A_prev: 上一层的激活函数输出， shape=(n_l-1, m)
    :param W: 当前层的权重矩阵, shape=(n_l, n_L-1)
    :param b: 当前层的权重矩阵, shape=(n_l, 1)
    :param activation: 激活函数的选择， "relu", "sigmoid"
    :return: 当前层的激活函数输出A, chache_forward = (cache_linear, cache_activation)
    """
    Z, cache_linear = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, cache_activation = sigmoid_forward(Z)
    elif activation == "relu":
        A, cache_activation = relu_forward(Z)
    return A, (cache_linear, cache_activation)


def L_model_forward(X, parameters):
    """
    多层前向传播
    :param X: 输入矩阵,shape=(n_0, m)
    :param parameters: python字典，里面存储着各层的参数
    :return: 1. AL， 最后一层的输出，y_hat
             2. python 列表，存储着各层的前向传播cache, c=;aches=[cache_forward_1,...........cache_forward_L]
             cache_forward_l = (cache_linear_l, cache_activation_l)
             cache_linear_l = (A_prev, W, b)
             cache_activation_l = Z
    """
    # 总层数L
    L = len(parameters) // 2
    # 创建一个列表，来存贮各个层的cache_forward
    caches = []
    A = X
    for l in range(1, L):   # relu 执行L-1次
        A_prev = A
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        A, cache_forward = linear_activation_forward(A_prev, W, b, "relu")
        caches.append(cache_forward)
    # sigmoid
    W = parameters["W" + str(L)]
    b = parameters["b" + str(L)]
    AL, cache_forward = linear_activation_forward(A, W, b, "sigmoid")
    caches.append(cache_forward)
    assert len(caches) == L
    return AL, caches


# 计算cost
def compute_cost(AL, Y):
    """
    计算交叉熵代价函数
    :param AL: 最后一层的激活函数输出
    :param Y:  标签
    :return: cost
    """
    assert AL.shape == Y.shape
    m = Y.shape[1]
    cost = -np.sum((Y * np.log(AL) + (1-Y) * np.log(1-AL))) / m
    cost = np.squeeze(cost)  # 确保cost是我想要的形式, np.squeeze([[cost]]) ---> cost
    return cost


# 反向传播:  1. single layer linear backward
#           2. single layer activation backward
#           3. single layer linear---->activation backward
#           4. multilayer backward


def linear_backward(dZ, cache_linear):
    """
    线性部分求导数，假设你已经知道了dZ, 求dW, db, dA_prev。 其中dA_prev是下一层反向传播的输入
    :param dZ: Z的导数
    :param cache_linear: linear前向传播, 存储着(A_prev, W, b)
    :return: dA_prev, dW, db
    """
    A_prev, W, b = cache_linear
    m = A_prev.shape[1]
    assert dZ.shape[0] == W.shape[0]
    dW = dZ.dot(A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = W.T.dot(dZ)
    assert dW.shape == W.shape
    assert db.shape == b.shape
    return dA_prev, dW, db


# 计算activation backward
def relu_backward(dA, cache_activation):
    """
    当激活函数为relu时：通过dA求dZ, dZ = dA * g'(Z)
    :param dA:A的导数
    :param cache_activation: activation前向传播, 存储着Z
    :return: dZ
    """
    Z = cache_activation
    assert dA.shape == Z.shape
    s = np.where(Z > 0, 1, 0)
    dZ = dA * s
    return dZ


def sigmoid_backward(dA, cache_activation):
    """
      当激活函数为sigmoid时：通过dA求dZ, dZ = dA * g'(Z)
      :param dA:A的导数
      :param cache_activation: activation前向传播, 存储着Z
      :return: dZ
      """
    Z = cache_activation
    assert dA.shape == Z.shape
    s = 1 / (1 + np.exp(-Z))
    s = s * (1 - s)
    dZ = dA * s
    return dZ


def linear_activation_backward(dA, cache_forward, activation):
    """
    singel layer： linear-->activation 合并，求出dA_prev, dW, db
    :param dA: 当前层A的导数
    :param cache_forward: 当前层对应的cache_forward = (cache_linear, cache_activation)
    :param activation: 激活函数的选择, "relu", "sigmoid"
    :return: dA_prev, dW, db
    """
    cache_linear, cache_activation = cache_forward
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, cache_activation)
        dA_prev, dW, db = linear_backward(dZ, cache_linear)
    elif activation == "relu":
        dZ = relu_backward(dA, cache_activation)
        dA_prev, dW, db = linear_backward(dZ, cache_linear)
    return dA_prev, dW, db


def L_model_backwrad(AL, Y, caches):
    """
    多层神经网络的反向传播
    :param AL: 最后一层的激活函数的输出
    :param Y: 标签
    :param caches: 1.反向传播的caches python列表, caches =[cache_forward_1,.....cache_forward_L]
                   2. cache_forward_l = (cache_linear, cache_activation)
                   3. cache_linear = (A_prev, W, b)
                   4. cache_activation = Z
    :return: 返回梯度grads, 是一个python字典， grads={dAL-1, dWL, dbL,......dA0, dW1, db1}
    """
    Y = Y.reshape(AL.shape)
    assert AL.shape == Y.shape
    # 总层数L
    L = len(caches)
    grads = {}
    # 反向传播的开头, 你需要dAL作为初始化的输入
    # 求出输出层的dA_prev,dW, db
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    dA_prev, dW, db = linear_activation_backward(dAL, caches[L-1], "sigmoid")
    grads["dA" + str(L-1)] = dA_prev
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db
    # 进行L-1次反向传播
    for l in reversed(range(1, L)):
        dA = grads["dA" + str(l)]
        dA_prev, dW, db = linear_activation_backward(dA, caches[l-1], "relu")
        grads["dA" + str(l-1)] = dA_prev
        grads["dW" + str(l)] = dW
        grads["db" + str(l)] = db
    return grads


# 更新参数
def update_parameters(parameters, grads, learning_rate):
    """
    更新参数
    :param parameters: 参数字典
    :param grads: 梯度字典
    :param learning_rate: 学习率
    :return: 更新之后的参数字典
    """
    # 获取层数
    L = len(parameters) // 2
    for l in range(1, L+1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - grads["dW" + str(l)] * learning_rate
        parameters["b" + str(l)] = parameters["b" + str(l)] - grads["db" + str(l)] * learning_rate
    return parameters


# 准确率
def accuracy(X, Y, parameters):
    m = X.shape[1]
    L = len(parameters) // 2
    if L==1:
        AL, cache_forward = linear_activation_forward(X, parameters["W1"], parameters["b1"], "sigmoid")
    else:
        AL, caches = L_model_forward(X, parameters)
    prediction = np.where(AL > 0.5, 1, 0)
    accuracy = np.sum(np.where(prediction==Y, 1, 0)) / m
    return accuracy


# 加载数据
def load_data():
    train_dataset = h5py.File('C:\\Git_Hub\\Logistic_regression\\datasets\\train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('C:\\Git_Hub\\Logistic_regression\\datasets\\test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
