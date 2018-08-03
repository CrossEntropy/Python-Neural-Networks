import numpy as np
import math
import sklearn.datasets


def initialize_parameters(layers_sizes):
    """
    初始化参数，采用He初始化方法
    :param layers_sizes: 从输入层到输出层节点数的列表
    :return: 参数字典
    """
    np.random.seed(3)  # Andrew Ng 选择seed=3
    L = len(layers_sizes)
    parameters = {}
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_sizes[l], layers_sizes[l-1])*\
                                   np.sqrt(2/layers_sizes[l-1])
        parameters["b" + str(l)] = np.zeros(shape=(layers_sizes[l], 1))
    return parameters


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A


def relu(Z):
    A = np.maximum(0, Z)
    return A


def linear_forward_prop(A_prev, W, b):
    """
    前向传播中的线性部分
    :param A_prev: 前一层的输出
    :param W: 当前层的权重
    :param b: 当前层的偏置
    :return: A, cache_linear = (A_prev, W) 方便于反向传播中的计算
    """
    Z = W.dot(A_prev) + b
    cache_linear = (A_prev, W)
    return Z, cache_linear


def linear_activation_forward_prop(A_prev, W, b, activation):
    """
    前向传播中的线性部分和激活函数部分整合起来，构造出单层的前向传播
    :param A_prev: 前一层的输出
    :param W: 当前层的权重
    :param b: 当前层的偏置
    :param activation: 激活函数的选择, "sigmoid" or "relu"
    :return: A, cache = (cache_linear, cache_activation) 方便于方向传播的计算
    """
    Z, cache_linear = linear_forward_prop(A_prev, W, b)
    if activation == "relu":
        A = relu(Z)
    elif activation == "sigmoid":
        A = sigmoid(Z)
    cache_activation = Z
    return A, (cache_linear, cache_activation)


def forward_propagation(X, parameters):
    """
    构建L层的前向传播模型
    :param X: 输入
    :param parameters: 参数字典
    :return: AL 和 caches=[cache_1, cache_2,.....cache_L]
    """
    L = len(parameters) // 2
    A = X
    caches = []
    # L-1层前向传播
    for l in range(1, L):
        A_prev = A
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        A, cache = linear_activation_forward_prop(A_prev, W, b, "relu")
        caches.append(cache)
    # 一层sigmoid输出
    W = parameters["W" + str(L)]
    b = parameters["b" + str(L)]
    AL, cache = linear_activation_forward_prop(A, W, b, "sigmoid")
    caches.append(cache)
    return AL, caches


def linear_backward_prop(dZ, cache_linear):
    """
    反向传播中的线性部分
    :param dZ: Z的梯度
    :param cache_linear: 前向传播中线性部分的cache_linear, cache_linear = (A_prev, W)
    :return: dA_prev, dW, db
    """
    m = dZ.shape[1]
    A_prev, W = cache_linear
    dW = dZ.dot(A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = W.T.dot(dZ)
    return dA_prev, dW, db


def activation_backward_prop(dA, cache_activation, activation):
    """
    反向传播中激活函数求导部分
    :param dA: A的梯度
    :param cache_activation: 线性传播中激活函数部分的cache_activation = Z
    :param activation: 激活函数的选择， "relu" or "sigmoid"
    :return: dZ
    """
    Z = cache_activation
    if activation == "relu":
        s = np.where(Z > 0, 1, 0)
        dZ = dA * s
    elif activation == "sigmoid":
        A = 1 / (1 + np.exp(-Z))
        dZ = dA * A * (1-A)
    return dZ


def activation_linear_backward_prop(dA, cache, activation):
    """
    反向传播中的线性部分和激活函数部分整合起来，构造出单层的反向传播
    :param dA: A的梯度
    :param cache: 前向传播的cache= (cache_linear, cache_activation)
    :param activation: 激活函数的选择, "sigmoid" or "relu"
    :return: dA_prev, dW, db
    """
    cache_linear, cache_activation = cache
    dZ = activation_backward_prop(dA, cache_activation, activation)
    dA_prev, dW, db = linear_backward_prop(dZ, cache_linear)
    return dA_prev, dW, db


def backward_propagation(AL, Y, caches):
    """
    构造L层的反向传播模型
    :param AL: L层前向传播模型的输出
    :param Y: 标签
    :param caches: [cache1, cache2,....cacheL]
    :return: grads, 一个字典，存储着梯度的字典
    """
    grads = {}
    # 初始化反向传播的输入
    L = len(caches)
    cache_L = caches[-1]
    cache_linear, cache_activation = cache_L
    dZL = AL - Y
    dA_prev, dW, db = linear_backward_prop(dZL, cache_linear)
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db
    # 执行剩余的L-1层的反向传播
    for l in reversed(range(1, L)):
        cache = caches[l-1]
        dA = dA_prev
        dA_prev, dW, db = activation_linear_backward_prop(dA, cache, activation="relu")
        grads["dW" + str(l)] = dW
        grads["db" + str(l)] = db
    return grads


# 计算损失函数
def compute_cost(AL, Y):
    m = AL.shape[1]
    logprobs = np.multiply(np.log(AL), -Y) + np.multiply(Y-1, np.log(1-AL))
    cost = np.nansum(logprobs) / m
    return cost


# 定义准确率函数
def prediction_accuracy(X, Y, parameters):
    AL, caches = forward_propagation(X, parameters)
    m = AL.shape[1]
    prediction = np.where(AL > 0.5, 1, 0)
    accuracy = np.where(prediction==Y, 1, 0)
    accuracy = np.sum(accuracy) / m
    return accuracy


# 不采用任何优化算法来跟新参数
def mini_batches_update_with_gd(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1 ,L+1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]
    return parameters


# 将数据集划分成一个一个的mini_batch
def random_mini_batches(X, Y, mini_batch_size, seed):
    np.random.seed(seed)
    mini_batches = []
    # 需要先将X,Y shuffled
    m = X.shape[1]
    permutation = np.random.permutation(m)
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[:, permutation]
    # 计算出mini_batch的个数
    num_complete = math.floor(m / mini_batch_size)  # 向下取整
    # 迭代取出每一个mini_bath，组成元组，放入到mini_batches
    for i in range(num_complete):
        X_batch = X_shuffled[:, i * mini_batch_size:(i+1)*mini_batch_size]
        Y_batch = Y_shuffled[:, i * mini_batch_size:(i+1)*mini_batch_size]
        mini_batch = (X_batch, Y_batch)
        mini_batches.append(mini_batch)
    # 很有可能 m 不能被batch_size整除
    if m % mini_batch_size !=0:
        X_batch = X_shuffled[:, num_complete*mini_batch_size:]
        Y_batch = Y_shuffled[:, num_complete*mini_batch_size:]
        mini_batch =(X_batch, Y_batch)
        mini_batches.append(mini_batch)
    return mini_batches


def initialize_velocity(parameters):
    # 初始化VdW
    L = len(parameters)//2
    V = {}
    for l in range(1, L+1):
        V["dW"+str(l)] = np.zeros(shape=parameters["W"+str(l)].shape)
        V["db"+str(l)] = np.zeros(shape=parameters["b"+str(l)].shape)
    return V


def initialize_Adam(parameters):
    # 初始化VdW
    # 初始化SdW
    L = len(parameters) // 2
    V = {}
    S = {}
    for l in range(1, L + 1):
        V["dW" + str(l)] = np.zeros(shape=parameters["W" + str(l)].shape)
        V["db" + str(l)] = np.zeros(shape=parameters["b" + str(l)].shape)
        S["dW" + str(l)] = np.zeros(shape=parameters["W" + str(l)].shape)
        S["db" + str(l)] = np.zeros(shape=parameters["b" + str(l)].shape)
    return V, S


def mini_batches_update_with_momentum(parameters, grads, V, learning_rate, beta=0.9):
    # 用momentum GD的时候，不需要使用偏差矫正
    L = len(parameters) // 2
    for l in range(1, L+1):
        V["dW" + str(l)] = beta * V["dW" + str(l)] + (1 - beta) * grads["dW" + str(l)]
        V["db" + str(l)] = beta * V["db" + str(l)] + (1 - beta) * grads["db" + str(l)]
    parameters = mini_batches_update_with_gd(parameters, V, learning_rate)
    return parameters, V


def mini_batches_update_with_Adam(parameters, grads, V, S, t, learning_rate, beta1=0.9, beta2=0.99,
                                  epsilon=1E-7):
    # 注意偏差矫正
    # 返回V, S是没有经过偏差矫正的
    # 更新参数使用的是经过偏差矫正过的V_correct, S_correct
    L = len(parameters) // 2
    V_correct = {}
    S_correct = {}
    # 先更新V, S, 并计算出V_correct, S_correct
    for l in range(1, L+1):
        V["dW" + str(l)] = beta1 * V["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
        V["db" + str(l)] = beta1 * V["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]
        S["dW" + str(l)] = beta2 * S["dW" + str(l)] + (1 - beta2) * (grads["dW" + str(l)] ** 2)
        S["db" + str(l)] = beta2 * S["db" + str(l)] + (1 - beta2) * (grads["db" + str(l)] ** 2)
        V_correct["dW" + str(l)] = V["dW" + str(l)] / (1 - beta1**t)
        V_correct["db" + str(l)] = V["db" + str(l)] / (1 - beta1**t)
        S_correct["dW" + str(l)] = S["dW" + str(l)] / (1 - beta2**t)
        S_correct["db" + str(l)] = S["db" + str(l)] / (1 - beta2**t)
    # 更新参数
    for l in range(1, L+1):
        parameters["W" + str(l)] -= learning_rate * (V_correct["dW" + str(l)]/
                                                     (np.sqrt(S_correct["dW" + str(l)])+epsilon))
        parameters["b" + str(l)] -= learning_rate * (V_correct["db" + str(l)]/
                                                     (np.sqrt(S_correct["db" + str(l)])+epsilon))
    return parameters, V, S


def load_dataset():
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2)  # 300 #0.2
    # Visualize the data
    # plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    return train_X, train_Y

