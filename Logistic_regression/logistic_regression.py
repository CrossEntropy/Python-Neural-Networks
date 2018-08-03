from Logistic_regression.tool_kit import *
import matplotlib.pyplot as plt


def logistic_model(X, Y, dims_layer, num_iterations=3500, learning_rate=0.0075, print_cost=False):
    costs = []
    m = X.shape[1]
    parameters = dict()
    parameters["W1"] = np.zeros(shape=(dims_layer[1], dims_layer[0]))
    parameters["b1"] = np.zeros(shape=(dims_layer[1], 1))
    for i in range(num_iterations):
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        # 前向传播
        A1, cache_activation = sigmoid_forward(W1.dot(X) + b1)
        # 计算cost
        cost = compute_cost(A1, Y)
        if i % 100 == 0 and print_cost:
            print("cost of {} iteration is {}".format(i, cost))
        if i % 100 == 0:
            costs.append(cost)
        # 反向传播
        dW1 = (A1-Y).dot(X.T) / m
        db1 = np.sum(A1-Y, axis=1, keepdims=True) / m
        # 更新参数
        parameters["W1"] = parameters["W1"] - dW1 * learning_rate
        parameters["b1"] = parameters["b1"] - db1 * learning_rate
    plt.plot(costs)
    plt.title("Logistic_regression regression, learning rate is "+str(learning_rate))
    plt.xlabel(r"$iteration (per hundreds)$")
    plt.ylabel(r"$cost$")
    plt.show()
    return parameters


if __name__ == "__main__":
    train_x_org, train_y, test_x_org, test_y, classes = load_data()
    train_x_flatten = train_x_org.reshape(train_x_org.shape[0], -1).T
    train_x = train_x_flatten / 255
    test_x_flatten = test_x_org.reshape(test_x_org.shape[0], -1).T
    test_x = test_x_flatten / 255
    dims_layer = [64*64*3, 1]
    parameters = logistic_model(train_x, train_y, dims_layer, print_cost=True)
    train_accuracy = accuracy(train_x, train_y, parameters)
    test_accuracy = accuracy(test_x, test_y, parameters)
    print("训练集的平均预测准确率为: " + str(train_accuracy * 100) + "%")
    print("测试集的平均预测准确率为: " + str(test_accuracy * 100) + "%")