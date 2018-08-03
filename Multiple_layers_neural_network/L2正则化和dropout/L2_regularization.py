from Multiple_layers_neural_network.L2正则化和dropout.reg_toolkit import *


def model(X, Y, layers_size, keep_prob=1, lambd=0, learning_rate=0.3, num_iterations=30000, print_cost=True):
    assert keep_prob == 1 or lambd == 0  # dropout 和 L2只能有一个执行
    # 初始化参数
    parameters = initialize_parameters_He(layers_size)
    costs = []
    for i in range(num_iterations):
        if lambd == 0 and keep_prob == 1:
            A3, caches = forward_propagation(X, parameters)
        elif lambd != 0:
            A3, caches = forward_propagation(X, parameters)
        else:
            A3, caches = forward_propagation_dropout(X, parameters, keep_prob)

        if lambd == 0 and keep_prob == 1:
            cost = compute_cost(A3, Y)
        elif lambd != 0:
            cost = compute_cost_L2(A3, Y, parameters, lambd)
        else:
            cost = compute_cost(A3, Y)

        if i % 1000 == 0 and print_cost:   # Andrew Ng是10000步记录一次，我这里用1000步记录一次
            costs.append(cost)
            print("第{}步的cost是{}".format(i, cost))
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(A3, Y, caches)
        elif lambd != 0:
            grads = backward_propagation_L2(A3, Y, caches, parameters, lambd)
        else:
            grads = backward_propagation_dropout(A3, Y, caches, keep_prob)
        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)
    plt.plot(costs)
    plt.title("learning rate = " + str(learning_rate))
    plt.xlabel("iterations (per thousands)")
    plt.ylabel("cost")
    plt.show()
    return parameters


if __name__ == "__main__":
    train_X, train_Y, test_X, test_Y = load_2D_datasets()
    parameters = model(train_X, train_Y, [2, 20, 3, 1])
    print("不加正则化, on the train: " + str(predict_accuracy(train_X, train_Y, parameters)))
    print("不加正则化, on the test: " + str(predict_accuracy(test_X, test_Y, parameters)))
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
    parameters = model(train_X, train_Y, [2, 20, 3, 1], lambd=0.7)
    print("L2正则化, on the train: " + str(predict_accuracy(train_X, train_Y, parameters)))
    print("L2正则化, on the test: " + str(predict_accuracy(test_X, test_Y, parameters)))
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
    parameters = model(train_X, train_Y, [2, 20, 3, 1],  keep_prob=0.86)
    print("dropout正则化, on the train: " + str(predict_accuracy(train_X, train_Y, parameters)))
    print("dropout正则化, on the test: " + str(predict_accuracy(test_X, test_Y, parameters)))
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


