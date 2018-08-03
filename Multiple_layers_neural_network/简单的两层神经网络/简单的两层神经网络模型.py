from Multiple_layers_neural_network.简单的两层神经网络.tool_kit import *
import matplotlib.pyplot as plt


def two_layer_model(X, Y, dims_layer, num_iterations=3500, learning_rate=0.075, print_cost=False):
    parameters = initialize_parameters(dims_layer)
    costs =[]
    for i in range(num_iterations):
        # 前向传播
        AL, caches = L_model_forward(X, parameters)
        # 计算cost
        cost = compute_cost(AL, Y)
        if i % 100 == 0 and print_cost:
            print("cost of {} iteration is {}".format(i, cost))
        if i % 100 == 0:
            costs.append(cost)
        # 反向传播
        grads = L_model_backwrad(AL, Y, caches)
        # 更新参数
        parameters =update_parameters(parameters, grads, learning_rate)
    plt.plot(costs)
    plt.title("Two layer, learning rate is " + str(learning_rate))
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
    dims_layer = [64 * 64 * 3, 7, 1]
    parameters = two_layer_model(train_x, train_y, dims_layer, learning_rate=0.0075, print_cost=True)
    train_accuracy = accuracy(train_x, train_y, parameters)
    test_accuracy = accuracy(test_x, test_y, parameters)
    print("训练集的平均预测准确率为: " + str(train_accuracy*100)+"%")
    print("测试集的平均预测准确率为: "+str(test_accuracy*100)+"%")


