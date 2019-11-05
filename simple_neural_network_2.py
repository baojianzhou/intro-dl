# -*- coding: utf-8 -*-

"""
This code is based on the following:
https://github.com/jldbc/numpy_neural_net/blob/master/four_layer_network.py
"""
import os
import numpy as np
from itertools import product
import matplotlib.pyplot as plt


def make_circles(n_samples=100, noise=None, factor=.8):
    if factor >= 1 or factor < 0:
        raise ValueError("'factor' has to be between 0 and 1.")
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    # so as not to have the first point = last point, we set endpoint=False
    line_space_out = np.linspace(0, 2 * np.pi, n_samples_out, endpoint=False)
    line_space_in = np.linspace(0, 2 * np.pi, n_samples_in, endpoint=False)
    outer_circle_x = np.cos(line_space_out)
    outer_circle_y = np.sin(line_space_out)
    inner_circle_x = np.cos(line_space_in) * factor
    inner_circle_y = np.sin(line_space_in) * factor
    x_tr = np.vstack((np.append(outer_circle_x, inner_circle_x), np.append(outer_circle_y, inner_circle_y))).T
    y_tr = np.hstack([np.zeros(n_samples_out, dtype=np.intp), np.ones(n_samples_in, dtype=np.intp)])
    if noise is not None:
        x_tr += np.random.normal(scale=noise, size=x_tr.shape)
    return x_tr, y_tr


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    return 1. * (x > 0)


def neural_network_create(input_dim, list_node_sizes, output_dim=2):
    model = dict()
    # first hidden layer
    model['W1'] = np.random.randn(input_dim, list_node_sizes[0])
    model['b1'] = np.zeros((1, list_node_sizes[0]))
    # second hidden layer
    model['W2'] = np.random.randn(list_node_sizes[0], list_node_sizes[1])
    model['b2'] = np.zeros((1, list_node_sizes[1]))
    # third hidden layer
    model['W3'] = np.random.randn(list_node_sizes[1], list_node_sizes[2])
    model['b3'] = np.zeros((1, list_node_sizes[2]))
    # fourth hidden layer
    model['W4'] = np.random.randn(list_node_sizes[2], list_node_sizes[3])
    model['b4'] = np.zeros((1, list_node_sizes[3]))
    # output layer
    model['W5'] = np.random.randn(list_node_sizes[3], output_dim)
    model['b5'] = np.zeros((1, output_dim))
    return model


def neural_network_feed_forward(model, x):
    # feed from input x to the 1st hidden layer
    z1 = x.dot(model['W1']) + model['b1']
    # activation of the first hidden layer
    a1 = relu(z1)
    # feed from the 1st layer to the 2nd hidden layer
    z2 = a1.dot(model['W2']) + model['b2']
    # activation of the 2nd hidden layer
    a2 = relu(z2)
    # feed from the 2nd layer to the 3rd hidden layer
    z3 = a2.dot(model['W3']) + model['b3']
    # activation of the 3rd hidden layer
    a3 = relu(z3)
    # feed from the 3rd layer to the 4th hidden layer
    z4 = a3.dot(model['W4']) + model['b4']
    # activation of the 4th hidden layer
    a4 = relu(z4)
    # feed from the 4th layer to the output layer
    z5 = a4.dot(model['W5']) + model['b5']
    exp_scores = np.exp(z5)
    out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return z1, a1, z2, a2, z3, a3, z4, a4, z5, out


def neural_network_back_propagation(model, x, y, a1, a2, a3, a4, output):
    delta5 = output
    delta5[range(x.shape[0]), y] -= 1  # y_pred - y

    dW5 = a4.T.dot(delta5)
    db5 = np.sum(delta5, axis=0, keepdims=True)
    delta4 = delta5.dot(model['W5'].T) * relu_derivative(a4)  # if ReLU

    dW4 = a3.T.dot(delta4)
    db4 = np.sum(delta4, axis=0, keepdims=True)
    delta3 = delta4.dot(model['W4'].T) * relu_derivative(a3)  # if ReLU

    dW3 = a2.T.dot(delta3)
    db3 = np.sum(delta3, axis=0, keepdims=True)
    delta2 = delta3.dot(model['W3'].T) * relu_derivative(a2)  # if ReLU
    dW2 = np.dot(a1.T, delta2)
    db2 = np.sum(delta2, axis=0)
    # delta2 = delta3.dot(model['W2'].T) * (1 - np.power(a1, 2)) #if tanh
    delta1 = delta2.dot(model['W2'].T) * relu_derivative(a1)  # if ReLU
    dW1 = np.dot(x.T, delta1)
    db1 = np.sum(delta1, axis=0)
    return dW1, dW2, dW3, dW4, dW5, db1, db2, db3, db4, db5


def calculate_loss(model, x_tr, y_tr):
    num_examples = x_tr.shape[0]
    z1, a1, z2, a2, z3, a3, z4, a4, z5, out = neural_network_feed_forward(model, x_tr)
    prob = out / np.sum(out, axis=1, keepdims=True)  # soft-max
    # https://deepnotes.io/softmax-crossentropy
    log_prob = -np.log(prob[range(num_examples), y_tr])  # the log likelihood
    return np.sum(log_prob) / num_examples


def save_contour(model, x_tr, y_tr, index, label):
    grid = np.mgrid[-1.5:1.5:100j, -1.5:1.5:100j]
    grid_2d = grid.reshape(2, -1).T
    _, _, _, _, _, _, _, _, _, output = neural_network_feed_forward(model, grid_2d)  # feed forward
    prediction_prob = output[:, 1]
    plt.figure(figsize=(10, 10))
    plt.title('Binary classification - epoch: %d' % index, fontsize=20)
    plt.xlabel('X', fontsize=15)
    plt.ylabel('Y', fontsize=15)
    plt.contourf(grid[0], grid[1], prediction_prob.reshape(100, 100), alpha=0.7)
    plt.scatter(x_tr[:, 0], x_tr[:, 1], c=y_tr.ravel(), s=50, edgecolors='black')
    if not os.path.exists("./figures"):
        os.mkdir("./figures")
    plt.savefig("./figures/%s_contour_index_%03d.png" % (label, index))
    plt.close()


def save_losses(losses, label):
    plt.figure(figsize=(10, 10))
    plt.title('Binary classification - losses', fontsize=20)
    plt.xlabel('Epochs', fontsize=15)
    plt.plot(losses, color='blue', linewidth=2.5)
    if not os.path.exists("./figures"):
        os.mkdir("./figures")
    plt.savefig("./figures/%s_losses.png" % label)
    plt.close()


def train_batch(model, x_tr, y_tr, num_passes, learning_rate):
    losses = []
    for epoch in range(num_passes):
        z1, a1, z2, a2, z3, a3, z4, a4, z5, out = neural_network_feed_forward(model, x_tr)  # feed forward
        pd = neural_network_back_propagation(model, x_tr, y_tr, a1, a2, a3, a4, out)
        dW1, dW2, dW3, dW4, dW5, db1, db2, db3, db4, db5 = pd
        # update weights and biases
        model['W1'] -= learning_rate * dW1
        model['b1'] -= learning_rate * db1
        model['W2'] -= learning_rate * dW2
        model['b2'] -= learning_rate * db2
        model['W3'] -= learning_rate * dW3
        model['b3'] -= learning_rate * db3
        model['W4'] -= learning_rate * dW4
        model['b4'] -= learning_rate * db4
        model['W5'] -= learning_rate * dW5
        model['b5'] -= learning_rate * db5

        losses.append(calculate_loss(model, x_tr, y_tr))
        save_contour(model, x_tr, y_tr, epoch, 'batch')
        print('epoch: %02d loss: %.4f' % (epoch, losses[-1]))
    save_losses(losses, 'batch')


def train_stochastic(model, x_tr, y_tr, num_passes, batch_size, learning_rate):
    losses = []
    for epoch, cur_iter in product(range(num_passes), range(len(x_tr) / batch_size)):
        start_ind, end_ind = cur_iter * batch_size, (cur_iter + 1) * batch_size
        cur_tr_x, cur_tr_y = x_tr[start_ind:end_ind], y_tr[start_ind:end_ind]
        z1, a1, z2, a2, z3, a3, z4, a4, z5, output = neural_network_feed_forward(model, cur_tr_x)
        pd = neural_network_back_propagation(model, cur_tr_x, cur_tr_y, a1, a2, a3, a4, output)
        dW1, dW2, dW3, dW4, dW5, db1, db2, db3, db4, db5 = pd
        # update weights and biases
        model['W1'] -= learning_rate * dW1
        model['b1'] -= learning_rate * db1
        model['W2'] -= learning_rate * dW2
        model['b2'] -= learning_rate * db2
        model['W3'] -= learning_rate * dW3
        model['b3'] -= learning_rate * db3
        model['W4'] -= learning_rate * dW4
        model['b4'] -= learning_rate * db4
        model['W5'] -= learning_rate * dW5
        model['b5'] -= learning_rate * db5
        if cur_iter == 0:
            losses.append(calculate_loss(model, x_tr, y_tr))
            save_contour(model, x_tr, y_tr, epoch, 'stochastic')
            print('epoch: %02d loss: %.4f' % (epoch, losses[-1]))
    save_losses(losses, 'stochastic')


def create_train_test_data(num_tr_samples=1000, num_te_samples=200):
    x_tr, y_tr = make_circles(n_samples=num_tr_samples, factor=.3, noise=.10)
    x_te, y_te = make_circles(n_samples=num_te_samples, factor=.3, noise=.10)
    rand_tr = np.random.permutation(len(x_tr))
    x_tr, y_tr = x_tr[rand_tr], y_tr[rand_tr]
    rand_te = np.random.permutation(len(x_te))
    x_te, y_te = x_te[rand_te], y_te[rand_te]
    return x_tr, y_tr, x_te, y_te


def test_batch(x_tr, y_tr, x_te, y_te):
    model = neural_network_create(input_dim=x_tr.shape[1], list_node_sizes=[4, 6, 6, 4], output_dim=2)
    train_batch(model, x_tr, y_tr, num_passes=50, learning_rate=0.001)
    output = neural_network_feed_forward(model, x_te)
    success = 0
    for ind, item in enumerate(output[-1]):
        if y_te[ind] == np.argmax(item):
            success += 1
    print(success / float(len(y_te)))


def test_stochastic(x_tr, y_tr, x_te, y_te):
    model = neural_network_create(input_dim=x_tr.shape[1], list_node_sizes=[4, 6, 6, 4], output_dim=2)
    train_stochastic(model=model, x_tr=x_tr, y_tr=y_tr, num_passes=50, batch_size=20, learning_rate=0.001)
    output = neural_network_feed_forward(model, x_te)
    success = 0
    for ind, item in enumerate(output[-1]):
        if y_te[ind] == np.argmax(item):
            success += 1
    print('test accuracy: %.4f' % (success / float(len(y_te))))


def main():
    x_tr, y_tr, x_te, y_te = create_train_test_data(num_tr_samples=1000, num_te_samples=200)
    plt.figure(figsize=(10, 10))
    plt.title('Dataset', fontsize=20)
    plt.xlabel('x1', fontsize=15)
    plt.ylabel('x2', fontsize=15)
    plt.scatter(x_tr[:, 0], x_tr[:, 1], c=y_tr.ravel(), s=50, edgecolors='black')
    plt.show()
    test_batch(x_tr, y_tr, x_te, y_te)
    test_stochastic(x_tr, y_tr, x_te, y_te)


if __name__ == "__main__":
    main()
