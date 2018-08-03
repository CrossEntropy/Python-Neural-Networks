import Multiple_layers_neural_network.梯度检验.gc_toolkit as tol_1
import Multiple_layers_neural_network.L2正则化和dropout.reg_toolkit as tol_2
import numpy as np


def gradinet_check(X, Y, parameters, gradients, epsilon=1E-7):
    parameters_vector, keys = tol_1.parameters_dictionary_to_vector(parameters)
    length = parameters_vector.shape[0]
    gradients_approx = np.zeros(shape=parameters_vector.shape)
    for i in range(length):
        theta_plus = np.copy(parameters_vector)
        theta_plus[i, 0] = theta_plus[i, 0] + epsilon
        theta = tol_1.parameters_vector_to_dictionary(theta_plus)
        A_plus, caches = tol_2.forward_propagation_dropout(X, theta, keep_prob=0.86)
        Cost_plus = tol_2.compute_cost(A_plus, Y)

        theta_minus = np.copy(parameters_vector)
        theta_minus[i, 0] = theta_minus[i, 0] - epsilon
        theta = tol_1.parameters_vector_to_dictionary(theta_minus)
        A_minus, caches = tol_2.forward_propagation_dropout(X, theta, keep_prob=0.86)
        Cost_minus = tol_2.compute_cost(A_minus, Y)

        numerator = Cost_plus - Cost_minus
        denominator = 2 * epsilon
        gradients_approx_i = numerator / denominator
        gradients_approx[i, 0]  = gradients_approx_i
    gradients = tol_1.gradient_dictionary_to_vector(gradients)

    # 计算误差
    numerator = np.linalg.norm(gradients_approx - gradients)
    denominator = np.linalg.norm(gradients) + np.linalg.norm(gradients_approx)
    difference = numerator / denominator

    if difference <= 1E-5:
        print("反向传播的程序是对的")
    else:
        print("反向传播的程序有问题")


if __name__ == "__main__":
    train_X, train_Y, test_X, test_Y = tol_2.load_2D_datasets()
    parameters = tol_2.initialize_parameters_He([2, 20, 3, 1])
    A3, caches = tol_2.forward_propagation_dropout(train_X, parameters, 0.86)
    grads = tol_2.backward_propagation_dropout(A3, train_Y, caches, 0.86)
    gradinet_check(train_X, train_Y, parameters, grads)


