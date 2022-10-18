import timeit
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(9, 7))
plt.axis([-30, 30, -30, 30])

# Import dataset.csv file

data = pd.read_csv("dataset.csv", index_col='Unnamed: 0')

# Setting intial Values and Parameter

features = data[['data_x', 'data_y']].to_numpy()
Label = data['Label'].to_numpy()
n_features = features.shape[1]
n_label = Label.ndim
n_node = 2
iteration = 1000
learning_rate = 0.02

W_h = np.random.randn(n_features, n_node)
W_o = np.random.randn(n_node, n_label)
b_h = np.zeros(n_node)
b_o = np.zeros(n_label)

# Define activate function


def sigmoid(x):
    y = 1/(1 + np.exp(-x))
    return y


def relu(x):
    if x >= 0:
        y = x
    else:
        y = 0
    return y

# Define Derivate function


def D_sigmoid(x):
    y = sigmoid(x) * (1-sigmoid(x))
    return y

# Forward_Propagation


def Forward_propagation(x, w, b, opt='sigmoid'):
    node_net = np.zeros(len(x))
    node_out = np.zeros(len(x))
    for i in range(len(features)):
        x_i = x[i, :]
        node_net[i] = np.matmul(x_i, w) + b
        if (opt == 'sigmoid'):
            node_out[i] = sigmoid(node_net[i])
        else:
            node_out[i] = relu(node_net[i])
    return node_out

# Back_Propagation


def Back_propagation(x, y, W_h, W_o, b_h, b_o, learn_rate=learning_rate, operate_out=True, opt='sigmoid'):
    # O_out = np.zeros(len(x))

    if operate_out == True:
        for i in range(len(x)):
            y_i = y[i]
            O_out = Forward_propagation(x, W_o, b_o, opt='sigmoid')
            W_o[0] = W_o[0] + learn_rate * \
                (y_i - O_out[i]) * O_out[i] * (1-O_out[i]) * x[i, 0]
            W_o[1] = W_o[1] + learn_rate * \
                (y_i - O_out[i]) * O_out[i] * (1-O_out[i]) * x[i, 1]
            b_o = b_o + learn_rate * \
                (y_i - O_out[i]) * O_out[i] * (1-O_out[i]) * 1
        return W_o, b_o

    else:
        for i in range(len(x)):
            y_i = y[i]
            h1_out = Forward_propagation(
                features, W_h[:, 0], b_h[0], opt='sigmoid').reshape((-1, 1))
            h2_out = Forward_propagation(
                features, W_h[:, 1], b_h[1], opt='sigmoid').reshape((-1, 1))
            h_out = np.concatenate((h1_out, h2_out), axis=1)
            O_out = Forward_propagation(h_out, W_o, b_o)

            W_h[0, 0] = W_h[0, 0] + learn_rate * \
                (y_i - O_out[i]) * O_out[i] * (1-O_out[i]) * W_o[0] * \
                h1_out[i] * (1 - h1_out[i]) * x[i, 0]
            W_h[1, 0] = W_h[1, 0] + learn_rate * \
                (y_i - O_out[i]) * O_out[i] * (1-O_out[i]) * W_o[0] * \
                h1_out[i] * (1 - h1_out[i]) * x[i, 1]
            W_h[0, 1] = W_h[0, 1] + learn_rate * \
                (y_i - O_out[i]) * O_out[i] * (1-O_out[i]) * W_o[1] * \
                h2_out[i] * (1 - h2_out[i]) * x[i, 0]
            W_h[1, 1] = W_h[1, 1] + learn_rate * \
                (y_i - O_out[i]) * O_out[i] * (1-O_out[i]) * W_o[1] * \
                h2_out[i] * (1 - h2_out[i]) * x[i, 1]
            b_h[0] = b_h[0] + learn_rate * \
                (y_i - O_out[i]) * O_out[i] * (1-O_out[i]) * W_o[0] * \
                h1_out[i] * (1 - h1_out[i]) * 1
            b_h[1] = b_h[1] + learn_rate * \
                (y_i - O_out[i]) * O_out[i] * (1-O_out[i]) * W_o[1] * \
                h2_out[i] * (1 - h2_out[i]) * 1

        return W_h, b_h


# Drawing intial line(before learning)

x_line = np.linspace(-30, 30, 100)
y_line1 = - W_h[0, 0] / W_h[1, 0] * x_line - b_h[0] / W_h[1, 0]
y_line2 = - W_h[0, 1] / W_h[1, 1] * x_line - b_h[1] / W_h[1, 1]
plt.plot(x_line, y_line1, '--k', label='initial line 1')
plt.plot(x_line, y_line2, '--k', label='initial line 2')
# print(f'Accuracy : {accuracy} %')

# Run whole batch
start_time = timeit.default_timer()  # Check initial time
for i in range(iteration):
    # Forward-Propagation
    h1_out = Forward_propagation(
        features, W_h[:, 0], b_h[0], opt='sigmoid').reshape((-1, 1))
    h2_out = Forward_propagation(
        features, W_h[:, 1], b_h[1], opt='sigmoid').reshape((-1, 1))
    node_h_out = np.concatenate((h1_out, h2_out), axis=1)
    node_o_out = Forward_propagation(node_h_out, W_o, b_o)

    # Back-Propagation
    W_o, b_o = Back_propagation(
        node_h_out, Label, 0, W_o, 0, b_o, operate_out=True)

    W_h, b_h = Back_propagation(
        features, Label, W_h, W_o, b_h, b_o, operate_out=False)
terminate_time = timeit.default_timer()  # Check final time
acting_time = round(terminate_time - start_time, 1)
print(f'{acting_time}초 걸렸습니다.')

# Accuracy

prediction = pd.Series(np.around(node_o_out))
data['prediction'] = prediction

accuracy = 0
for i in range(data.shape[0]):
    c1 = data["Label"].iloc[i]
    c2 = data['prediction'].iloc[i]
    if c1 == c2:
        accuracy += 1
print(f'Accuracy : {accuracy} %')

# Visualization

sns.scatterplot(data=data, x="data_x", y="data_y",
                hue="hue", hue_order=["GroupA", "GroupB"],
                style="hue")
y_line3 = - W_h[0, 0] / W_h[1, 0] * x_line - b_h[0] / W_h[1, 0]
y_line4 = - W_h[0, 1] / W_h[1, 1] * x_line - b_h[1] / W_h[1, 1]
plt.plot(x_line, y_line3, 'r', label='final line 1')
plt.plot(x_line, y_line4, 'r', label='final line 2')
plt.title('Sigmoid_2Layer')
plt.text(-25, -25, 'accuracy:' + str(accuracy) + ' %')
plt.show()
plt.savefig('./savefig_default.png')
