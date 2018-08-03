from Multiple_layers_neural_network.权重初始化对网络的影响.tool_kit import *
import matplotlib.pyplot as plt


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