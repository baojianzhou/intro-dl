# -*- coding: utf-8 -*-

"""

A Step by Step Back propagation Example.

This code is based on the implementation by Matt Mazur.
Tutorial: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
Github: https://github.com/mattm/simple-neural-network
"""
import numpy as np


def neuron_create(num_inputs, activation, bias, layer_id, neuron_id):
    """
    Create a neuron.
    :param num_inputs: input dimension
    :param activation: activation function.
    :param bias: bias for this activation.
    :param layer_id:
    :param neuron_id:
    :return:
    """
    return {'weights': np.zeros(num_inputs),
            'inputs': np.zeros(num_inputs),
            'bias': np.random.random() if bias is None else bias,
            'output': 0.0,
            'activation': activation,
            'layer_id': layer_id,
            'neuron_id': neuron_id}


def neuron_cal_output(neuron, inputs=None):
    """
    Given the neuron[inputs, weights, and a bias], it calculates the output of this neuron.
    :param inputs
    :param neuron:
    :return:
    """
    # update the inputs
    if inputs is not None:
        neuron['inputs'] = np.asarray(inputs)
    total_net_input = np.dot(neuron['inputs'], neuron['weights']) + neuron['bias']
    if neuron['activation'] == 'sigmoid':
        neuron['output'] = 1. / (1. + np.exp(-total_net_input))
        return neuron['output']
    elif neuron['activation'] == 'tanh':
        neuron['output'] = 1. / (1. + np.exp(-total_net_input))
        return neuron['output']
    elif neuron['activation'] == 'linear':
        neuron['output'] = total_net_input
        return neuron['output']
    else:
        return neuron['output']


def neuron_cal_error(neuron, target_output, loss='least_square'):
    """
    Given a target value, it calculates the error of this neuron.
    :param neuron
    :param target_output
    :param loss:
    :return:
    """
    pred_output = neuron_cal_output(neuron)
    if loss == 'least_square':
        return 0.5 * (target_output - pred_output) ** 2
    else:
        return 0.5 * (target_output - pred_output) ** 2


def neuron_cal_pd_error_wrt_total_net_input(neuron, target_output):
    """
    # Determine how much the neuron's total input has to change to move closer to the expected output
    # Now that we have the partial derivative of the error with respect to the output (∂E/∂yⱼ) and
    # the derivative of the output with respect to the total net input (dyⱼ/dzⱼ) we can calculate
    # the partial derivative of the error with respect to the total net input.
    # This value is also known as the delta (δ) [1]
    # δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    :return:
    """
    output = neuron_cal_output(neuron)
    # The partial derivative of the error with respect to actual output then is calculated by:
    # = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
    # = -(target output - actual output). The Wikipedia article on backpropagation [1] simplifies to
    # the following, but most other learning material does not [2]
    # = actual output - target output. Alternative, you can use (target - output), but then need to add it during
    # backpropagation [3]. Note that the actual output of the output neuron is often written as yⱼ and target output
    # as tⱼ so: = ∂E/∂yⱼ = -(tⱼ - yⱼ)
    pd_error_wrt_output = -(target_output - output)
    # The total net input into the neuron is squashed using logistic function to calculate the neuron's output:
    # yⱼ = φ = 1 / (1 + e^(-zⱼ)). Note that where ⱼ represents the output of the neurons in whatever layer we're
    # looking at and ᵢ represents the layer below it. The derivative (not partial derivative since there is only
    # one variable) of the output then is: dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
    pd_total_net_input_wrt_input = output * (1. - output)
    return pd_error_wrt_output * pd_total_net_input_wrt_input


def neuron_cal_pd_total_net_input_wrt_input(neuron):
    """
    The derivative of the sigmoid function.
    :param neuron: a sigmoid neuron.
    :return:
    """
    return neuron['output'] * (1. - neuron['output'])


def neuron_cal_pd_total_net_input_wrt_weight(neuron, index):
    """
    # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
    # = zⱼ = netⱼ = x₁w₁ + x₂w₂ .... The partial derivative of the total net input with respective to a given weight
    # (with everything else held constant) then is: = ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
    :return:
    """
    return neuron['inputs'][index]


def neural_network_create(num_inputs, num_hidden, num_outputs,
                          hidden_layer_weights=None, hidden_layer_bias=None,
                          output_layer_weights=None, output_layer_bias=None):
    hidden_layer = {'num_neurons': num_hidden,
                    'neurons': [neuron_create(num_inputs, 'sigmoid', hidden_layer_bias, 1, _)
                                for _ in range(num_hidden)]}
    output_layer = {'num_neurons': num_outputs,
                    'neurons': [neuron_create(num_hidden, 'sigmoid', output_layer_bias, 2, _)
                                for _ in range(num_outputs)]}
    # initialize weights from inputs to hidden layer neurons
    weight_num = 0
    for h in range(hidden_layer['num_neurons']):
        for i in range(num_inputs):
            if not hidden_layer_weights:
                hidden_layer['neurons'][h]['weights'][i] = np.random.random()
            else:
                hidden_layer['neurons'][h]['weights'][i] = hidden_layer_weights[weight_num]
            weight_num += 1
    # initialize weights from hidden layer neurons to output layer neurons
    weight_num = 0
    for o in range(output_layer['num_neurons']):
        for h in range(hidden_layer['num_neurons']):
            if not output_layer_weights:
                output_layer['neurons'][o]['weights'][h] = np.random.random()
            else:
                output_layer['neurons'][o]['weights'][h] = output_layer_weights[weight_num]
            weight_num += 1
    neural_network = {'num_inputs': num_inputs,
                      'num_hidden': num_hidden,
                      'num_outputs': num_outputs,
                      'hidden_layer': hidden_layer,
                      'output_layer': output_layer}
    print('created a new neural networks and finished the initialization ...')
    return neural_network


def neural_network_feed_forward(neural_network, train_xi):
    """
    Calculate the predicted value by this neural network.
    :param neural_network:
    :param train_xi:
    :return:
    """
    hidden_layer = neural_network['hidden_layer']
    output_layer = neural_network['output_layer']
    # hidden layer feeds forward activation.
    hidden_layer_outputs = np.zeros(hidden_layer['num_neurons'])
    for index, neuron in enumerate(hidden_layer['neurons']):
        hidden_layer_outputs[index] = neuron_cal_output(neuron, train_xi)
    # output layer feeds forward activation.
    outputs = np.zeros(hidden_layer['num_neurons'])
    for index, neuron in enumerate(output_layer['neurons']):
        outputs[index] = neuron_cal_output(neuron, hidden_layer_outputs)
    # final prediction
    return outputs


def neural_network_train(neural_network, train_x, train_y, learning_rate):
    """
    The training process of this neural network.
    :param neural_network:
    :param train_x:
    :param train_y:
    :param learning_rate:
    :return:
    """
    neural_network_feed_forward(neural_network, train_x)
    # 1. Output neuron deltas
    output_layer = neural_network['output_layer']
    pd_errors_wrt_output_neuron_total_net_input = np.zeros(output_layer['num_neurons'])
    for o in range(output_layer['num_neurons']):  # ∂E/∂zⱼ
        pd = neuron_cal_pd_error_wrt_total_net_input(output_layer['neurons'][o], target_output=train_y[o])
        pd_errors_wrt_output_neuron_total_net_input[o] = pd
    # 2. Hidden neuron deltas
    hidden_layer = neural_network['hidden_layer']
    pd_errors_wrt_hidden_neuron_total_net_input = np.zeros(hidden_layer['num_neurons'])
    for h in range(hidden_layer['num_neurons']):
        # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
        # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
        d_error_wrt_hidden_neuron_output = 0
        for o in range(output_layer['num_neurons']):
            d = pd_errors_wrt_output_neuron_total_net_input[o] * output_layer['neurons'][o]['weights'][h]
            d_error_wrt_hidden_neuron_output += d
        # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
        pd = d_error_wrt_hidden_neuron_output * neuron_cal_pd_total_net_input_wrt_input(hidden_layer['neurons'][h])
        pd_errors_wrt_hidden_neuron_total_net_input[h] = pd
    # 3. Update output neuron weights
    for o in range(output_layer['num_neurons']):
        for w_ho in range(len(output_layer['neurons'][o]['weights'])):
            # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
            pd = neuron_cal_pd_total_net_input_wrt_weight(output_layer['neurons'][o], w_ho)
            pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * pd
            # Δw = α * ∂Eⱼ/∂wᵢ
            output_layer['neurons'][o]['weights'][w_ho] -= learning_rate * pd_error_wrt_weight
    # 4. Update hidden neuron weights
    for h in range(hidden_layer['num_neurons']):
        for w_ih in range(len(hidden_layer['neurons'][h]['weights'])):
            # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
            pd = neuron_cal_pd_total_net_input_wrt_weight(hidden_layer['neurons'][h], w_ih)
            pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * pd
            # Δw = α * ∂Eⱼ/∂wᵢ
            hidden_layer['neurons'][h]['weights'][w_ih] -= learning_rate * pd_error_wrt_weight


def neural_network_total_error(neural_network, train_x, train_y):
    """
    The predicted error of this neural network.
    :param neural_network:
    :param train_x:
    :param train_y:
    :return:
    """
    assert len(train_x) == len(train_y)
    total_error = 0
    for t in range(len(train_x)):
        output = neural_network_feed_forward(neural_network, train_x[t])
        output_layer = neural_network['output_layer']
        for o in range(len(train_y[t])):
            total_error += neuron_cal_error(output_layer['neurons'][o], train_y[t][o])
    return total_error


def test_example_1():
    """
    This is a regression task. This test is the original example.
    Here we have a 2x2 neural network.
    :return:
    """
    num_inputs = 2
    num_hidden = 2
    num_outputs = 2
    hidden_layer_weights = [0.15, 0.2, 0.25, 0.3]
    hidden_layer_bias = 0.35
    output_layer_weights = [0.4, 0.45, 0.5, 0.55]
    output_layer_bias = 0.6
    nn = neural_network_create(num_inputs=num_inputs, num_hidden=num_hidden, num_outputs=num_outputs,
                               hidden_layer_weights=hidden_layer_weights, hidden_layer_bias=hidden_layer_bias,
                               output_layer_weights=output_layer_weights, output_layer_bias=output_layer_bias)
    train_losses = []
    learning_rate = 0.5
    train_x = [[0.05, 0.1]]
    train_y = [[0.01, 0.99]]
    for i in range(1000):
        train_xi, train_yi = train_x[0], train_y[0]
        neural_network_train(nn, train_x=train_xi, train_y=train_yi, learning_rate=learning_rate)
        if i % 50 == 0:
            cur_loss = neural_network_total_error(nn, train_x=train_x, train_y=train_y)
            train_losses.append(cur_loss)
            print(i, round(train_losses[-1], 9))
    import matplotlib.pyplot as plt
    plt.plot(train_losses)
    plt.show()
    return np.asarray(train_losses)


def test_example_2():
    """
    This is a regression task. This test is the example provided in our slides.
    Here we have a 3x2 neural network.
    :return:
    """
    num_inputs = 2
    num_hidden = 3
    num_outputs = 2
    hidden_layer_weights = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    hidden_layer_bias = 0.45
    output_layer_weights = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    output_layer_bias = 0.8
    nn = neural_network_create(num_inputs=num_inputs, num_hidden=num_hidden, num_outputs=num_outputs,
                               hidden_layer_weights=hidden_layer_weights, hidden_layer_bias=hidden_layer_bias,
                               output_layer_weights=output_layer_weights, output_layer_bias=output_layer_bias)
    train_losses = []
    train_x = [[0.05, 0.1]]
    train_y = [[0.01, 0.99]]
    for i in range(1000):
        train_xi, train_yi = train_x[0], train_y[0]
        neural_network_train(nn, train_x=train_xi, train_y=train_yi, learning_rate=.5)
        if i % 20 == 0:
            cur_loss = neural_network_total_error(nn, train_x=train_x, train_y=train_y)
            train_losses.append((i, cur_loss))
            print(i, train_losses[-1])
    import matplotlib.pyplot as plt
    plt.plot([_[0] for _ in train_losses], [_[1] for _ in train_losses],
             label='training loss', color='green', linewidth=2.)
    plt.title('Training losses of a 2x3x2 neural network', fontsize=20.)
    plt.xlabel('iteration', fontsize=20.)
    plt.ylabel('Least Square Loss', fontsize=20.)
    plt.show()
    return np.asarray(train_losses)


def test_example_3():
    """
    This is a classification task. This test is the example provided in the original code.
    Here we have a 5x1 neural network.
    The training examples are the XOR operations.
    x1 x2 y
    0  0  0
    0  1  1
    1  0  1
    1  1  0
    :return:
    """
    num_inputs = 2
    num_hidden = 5
    num_outputs = 1
    nn = neural_network_create(num_inputs=num_inputs, num_hidden=num_hidden, num_outputs=num_outputs)
    train_losses = []
    learning_rate = 0.5
    train_x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    train_y = [[0], [1], [1], [0]]
    for i in range(10000):
        rand_tr_index = np.random.randint(0, 4)
        train_xi, train_yi = train_x[rand_tr_index], train_y[rand_tr_index]
        neural_network_train(nn, train_x=train_xi, train_y=train_yi, learning_rate=learning_rate)
        if i % 50 == 0:
            cur_loss = neural_network_total_error(nn, train_x=train_x, train_y=train_y)
            train_losses.append(cur_loss)
            print(i, round(train_losses[-1], 9))
    import matplotlib.pyplot as plt
    plt.plot(train_losses)
    plt.show()
    return np.asarray(train_losses)


if __name__ == '__main__':
    test_example_2()
