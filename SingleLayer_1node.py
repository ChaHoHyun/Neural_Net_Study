from matplotlib import colors
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit
plt.figure(figsize=(9, 7))
plt.axis([-30, 30, -30, 30])

# import dataset.csv and processing Label

data = pd.read_csv("dataset.csv", index_col="Unnamed: 0")

# Activation functon


def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return(y)


def relu(x):
    if x >= 0:
        y = x
    else:
        y = 0
    return y


def step(x):
    if x >= 0:
        y = 1
    else:
        y = 0
    return y

# Define 'sigmoid', 'relu' derivate function


def der_sigmoid(x):
    y = sigmoid(x) * (1 - sigmoid(x))
    return y


def der_relu(x):
    if x >= 0:
        y = 1
    else:
        y = 0
    return y

# Inital Value and Parameter


features = data[["data_x", "data_y"]]
Label = data["Label"]

n_feature = features.shape[1]
n_output = 1
iteration = 1000
learning_rate = 0.01

W = np.random.rand(n_feature, 1)
b = np.zeros(n_output)

# ForwardPropagation


def forward_propagtion(x, w, b, opt='sigmoid'):
    net_out = np.zeros(x.shape[0])
    function_out = np.zeros(x.shape[0])
    for i in range(data.shape[0]):
        x_i = x.iloc[i]
        net_out[i] = (np.matmul(x_i, w) + b)

        if (opt == 'sigmoid'):
            function_out[i] = sigmoid(net_out[i])
        elif (opt == 'relu'):
            function_out[i] = relu(net_out[i])

    return function_out

# Backpropagation


def back_propagation(x, y, w, b, learn_rate=learning_rate, opt='sigmoid'):
    net_out = np.zeros(x.shape[0])
    function_out = np.zeros(x.shape[0])

    for i in range(Label.shape[0]):
        x_i = x.iloc[i]
        net_out[i] = np.matmul(x_i, w) + b
        function_out[i] = sigmoid(net_out[i])

        if opt == 'sigmoid':
            w[0] = w[0] + learning_rate * (y.iloc[i] - function_out[i]) * function_out[i] * \
                (1-function_out[i]) * x_i[0]
            w[1] = w[1] + learning_rate * (y.iloc[i] - function_out[i]) * function_out[i] * \
                (1-function_out[i]) * x_i[1]
            b = b + learning_rate * (y.iloc[i] - function_out[i]) * function_out[i] * \
                (1-function_out[i]) * 1

        elif opt == 'relu':
            w[0] = w[0] + learning_rate * \
                (y.iloc[i] - function_out[i]) * der_relu(net_out[i]) * x_i[0]
            w[1] = w[1] + learning_rate * \
                (y.iloc[i] - function_out[i]) * der_relu(net_out[i]) * x_i[1]
            b = b + learning_rate * \
                (y.iloc[i] - function_out[i]) * der_relu(net_out[i]) * 1

    return w, b

# Initial line


line_x = np.linspace(-30, 30, 100)
line_y = -W[0] / W[1] * line_x - b / W[1]
plt.plot(line_x, line_y, '--k', label='initial line')

# Run whole batch
start_time = timeit.default_timer()
for i in range(iteration):
    activate_out = forward_propagtion(features, W, b, opt='sigmoid')
    W, b = back_propagation(features, Label, W, b, opt='sigmoid')
end_time = timeit.default_timer()

operating_time = round(end_time - start_time, 1)
print(f"{operating_time}초 걸렸습니다.")

# Accuracy

prediction = pd.Series(np.around(activate_out))
data["prediction"] = prediction

accuracy = 0
for i in range(data.shape[0]):
    if data["Label"].iloc[i] == data['prediction'].iloc[i]:
        accuracy += 1

sns.scatterplot(data=data, x="data_x", y="data_y",
                hue="hue", hue_order=["GroupA", "GroupB"],
                style="hue")

print(f'Accuracy : {accuracy} %')

# Visualization

line_x = np.linspace(-30, 30, 100)
line_y = -W[0] / W[1] * line_x - b / W[1]
plt.plot(line_x, line_y, 'r', label='final line')

plt.title('Sigmoid_1 node')
plt.text(-25, -25, 'accuracy:' + str(accuracy) + ' %')
plt.show()
