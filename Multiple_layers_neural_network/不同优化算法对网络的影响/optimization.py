from Multiple_layers_neural_network.不同优化算法对网络的影响.opt_toolkit import *
import matplotlib.pyplot as plt
import time


def model(X, Y, layers_size, optimizer, num_epochs=10000, learning_rate=0.0007, mini_batch_size=64,
          beta=0.9, beta1=0.9, beta2=0.999, epsilon=1E-8, print_cost=True):
    seed = 0
    t = 0
    costs = []
    # 初始化参数，V, S
    parameters = initialize_parameters(layers_size)
    if optimizer == "momentum":
        V = initialize_velocity(parameters)
    elif optimizer == "Adam":
        V, S = initialize_Adam(parameters)
    elif optimizer == "gd":
        pass

    for i in range(num_epochs):
        # 每一次epoch都重新组合出一个新的mini_batches
        seed += 1
        mini_batches = random_mini_batches(X, Y, mini_batch_size, seed)
        for mini_batch in mini_batches:
            (mini_batch_X, mini_batch_Y) = mini_batch
            # 前向传播
            AL, caches = forward_propagation(mini_batch_X, parameters)
            # 计算cost
            cost = compute_cost(AL, mini_batch_Y)
            # 反向传播
            grads = backward_propagation(AL, mini_batch_Y, caches)
            # 更新参数
            if optimizer == "Adam":
                t += 1
                (parameters, V, S) = mini_batches_update_with_Adam(parameters, grads, V=V, S=S, beta1=beta1,
                                                               beta2=beta2, t=t, epsilon=epsilon, learning_rate=learning_rate)
            elif optimizer == "momentum":
                parameters, V = mini_batches_update_with_momentum(parameters, grads, V=V, beta=beta, learning_rate=learning_rate)
            elif optimizer == "gd":
                parameters = mini_batches_update_with_gd(parameters, grads, learning_rate)
        # 打印costs在epoch循环下
        if print_cost and i % 1000 == 0:
            # Andrew Ng epoch=1000 打印一次cost
            print("Cost after epoch {}: {}".format(i, cost))
        if i % 100 == 0:
            costs.append(cost)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()
    return parameters


def batch_model(X, Y, layers_size, num_iterations=10000000, learning_rate=0.0007):
    parameters = initialize_parameters(layers_size)
    for i in range(num_iterations):
        AL, caches = forward_propagation(X, parameters)
        accuracy = prediction_accuracy(X, Y, parameters)
        error = np.abs(accuracy - 0.94)
        if error <= 0.005:
            return parameters
        grads = backward_propagation(AL, Y, caches)
        parameters = mini_batches_update_with_gd(parameters, grads, learning_rate)
    return parameters


if __name__ == "__main__":
    """
    两组实验, 第一组实验与coursera相同，第二组实验自己做的，batch与mini_batch相比较时间
    """
    a = 2
    if a == 1:
        train_X, train_Y = load_dataset()
        # 数据 normalization
        # u = np.mean(train_X, axis=1, keepdims=True)
        # gamma = np.std(train_X, axis=1, keepdims=True)
        # train_X = (train_X - u) / gamma
        """
        将数据normalization, 在相同的迭代epoch下， 提高了准确率，减少了震荡
        """
        print("不采用优化算法的mini_batches")
        parameters = model(train_X, train_Y, [train_X.shape[0], 5, 2, 1], optimizer="gd")
        print("预测的准确率: "+str(prediction_accuracy(train_X, train_Y, parameters)))
        print("采用Momentum下降的mini_batches")
        parameters = model(train_X, train_Y, [train_X.shape[0], 5, 2, 1], optimizer="momentum")
        print("预测的准确率: " + str(prediction_accuracy(train_X, train_Y, parameters)))
        print("采用Adam梯度下降的mini_batches")
        parameters = model(train_X, train_Y, [train_X.shape[0], 5, 2, 1], optimizer="Adam")
        print("预测的准确率: " + str(prediction_accuracy(train_X, train_Y, parameters)))
    else:
        """
        Adam 达到94%的准确率只需要20s
        batch 达到93.6%的准确率需要65s, 差距明显
        """
        train_X, train_Y = load_dataset()
        print("采用Adam梯度下降的mini_batches")
        start = time.time()
        parameters = model(train_X, train_Y, [train_X.shape[0], 5, 2, 1], optimizer="Adam")
        end = time.time()
        print("预测的准确率: " + str(prediction_accuracy(train_X, train_Y, parameters)))
        print("时间为：" + str(end-start) + "s")

        print("batch梯度下降")
        start = time.time()
        parameters = batch_model(train_X, train_Y, [train_X.shape[0], 5, 2, 1])
        end = time.time()
        print("预测的准确率: " + str(prediction_accuracy(train_X, train_Y, parameters)))
        print("时间为：" + str(end - start) + "s")