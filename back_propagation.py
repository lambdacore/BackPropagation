# Jonathan Harrington

import pandas as pd
import numpy as np


# return a dic which contains df of a rdm gen weights between -0.05 and 0.05
def gen_rdm_weights(input_num: int, hidden_num: int, output_num: int):
    weights = {"input": pd.DataFrame((np.random.rand(input_num + 1, hidden_num) - 0.5) * 0.1),
               "hidden": pd.DataFrame((np.random.rand(hidden_num + 1, output_num) - 0.5) * 0.1)}
    return weights


# return a dic that contains outputs of the hidden / output layer
def fwrd_pass(inputs: list, weights: pd.DataFrame):
    outputs = {}
    # ignor matmul error
    hidden_outputs = list(activation_function(inputs @ weights["input"]))
    outputs["hidden"] = hidden_outputs.copy()
    hidden_outputs.insert(0, 1.0)
    # ignor matmul error
    outputs["output"] = list(activation_function(hidden_outputs @ weights["hidden"]))
    return outputs


# takes in the input / out / target / lr and then returns a float
def bwrd_pass(inputs: list, outputs: list, target_outputs: list, learning_rate: float, weights: pd.DataFrame):

    # calc the error in hidden / output layer
    errors = {"output": [output * (1 - output) for output in outputs["output"]] * (np.array(target_outputs) - np.array(outputs["output"]))}

    errors["hidden"] = [output * (1 - output) for output in outputs["hidden"]] * np.array(weights["hidden"] @ errors["output"])[1:]

    # update input weights
    for index in range(len(weights["input"].columns)):
        weights["input"][index] += learning_rate * errors["hidden"][index] * pd.Series(inputs)

    # update hidden weights
    hidden_outputs = outputs["hidden"].copy()
    hidden_outputs.insert(0, 1.0)
    for index in range(len(weights["hidden"].columns)):
        weights["hidden"][index] += learning_rate * errors["output"][index] * pd.Series(hidden_outputs)

    # we then return the instance errors
    return sum(0.5 * (np.array(target_outputs) - np.array(outputs["output"])) ** 2)


# activater / logistic function
def activation_function(network_value):
    return 1 / (1 + np.e ** -network_value)
