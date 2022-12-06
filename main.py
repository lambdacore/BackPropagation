# Jonathan Harrington 12/04/22

import pandas as pd
import time
from back_propagation import gen_rdm_weights, fwrd_pass, bwrd_pass

def start_prop():
    num_input_nodes = 784  # 784 given
    num_hidden_nodes = 50  # 16
    num_output_nodes = 10  # because we have 10 different digits
    lr = 0.1  # learning rate
    epoch = 10 # 10
    training_set = 5000 # 5000 / 60,000
    testing_set = 10000

    # used row major order
    # Training 60,000
    input_df = pd.read_csv("data/training60000.csv", header=None)
    output_df = pd.read_csv("data/training60000_labels.csv", header=None)
    gen_weights = gen_rdm_weights(num_input_nodes, num_hidden_nodes, num_output_nodes)
    print("Building Model")

    # build model until convergence

    for epoch in range(epoch):
        network_error = 0  # default value
        start = time.time()
        for i in range(training_set):
            # input and output are updated
            inputs = list(input_df.iloc[i])
            inputs.insert(0, 1.0)
            target_outputs = [0.01] * num_output_nodes  # 10 * 0.01 = 0.1
            target_outputs[next(iter(output_df.iloc[i]))] = 0.99  # this is what we are shooting for

            # outputs and update weights are set
            predicted_outputs = fwrd_pass(inputs, gen_weights)
            network_error += bwrd_pass(inputs, predicted_outputs, target_outputs, lr, gen_weights)
        print("End of Epoch {0}:".format(epoch + 1), network_error)
        end = time.time()
        print((end - start) / 60.00, " min")

    # Testing 10,000
    input_df = pd.read_csv("data/testing10000.csv", header=None)
    output_df = pd.read_csv("data/testing10000_labels.csv", header=None)

    print("===== Model Created =====")
    print()
    print("===== Testing Beginning =====")
    print()
    correct_clfy = 0
    for i in range(testing_set):
        inputs = list(input_df.iloc[i])
        inputs.insert(0, 1.0)
        target_label = next(iter(output_df.iloc[i]))
        outputs = fwrd_pass(inputs, gen_weights)

        # If target output equals output
        if target_label == outputs["output"].index(max(outputs["output"])):
            correct_clfy += 1

    #  results of test
    print("========= Results ===========")
    print("network properties: Input: {0}, Hidden: {1}, Output: {2}".format(num_input_nodes, num_hidden_nodes, num_output_nodes))
    print("learning rate: {0}, Epoch: {1}".format(lr, epoch+1))
    print("correct classification = ", correct_clfy)
    print("incorrect classification = ", testing_set - correct_clfy)
    print("accuracy = ", (correct_clfy / testing_set) * 100, "%")

if __name__ == '__main__':
    start_prop()
