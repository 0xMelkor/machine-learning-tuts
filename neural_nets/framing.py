"""
      _             _                              _                   
     | |           | |                            | |                  
  ___| |_ __ _  ___| | __  ___ _ __ ___   __ _ ___| |__   ___ _ __ ___ 
 / __| __/ _` |/ __| |/ / / __| '_ ` _ \ / _` / __| '_ \ / _ \ '__/ __|
 \__ \ || (_| | (__|   <  \__ \ | | | | | (_| \__ \ | | |  __/ |  \__ \
 |___/\__\__,_|\___|_|\_\ |___/_| |_| |_|\__,_|___/_| |_|\___|_|  |___/

@author: Andrea Simeoni 21 ott 2017   
https://github.com/insanediv/machine-learning-tuts/blob/master/neural_nets/framing.py
"""
import numpy as np


def frame_data(dataset, time_steps=1, dropnan=True):
    """
    Data framing inspired to
    https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
    :param dataset: set of sequenced data
    :param time_steps: how many timesteps to look backward
    :param dropnan: because of shifting some rows have None objects. Set true to remove such rows
    :return: 
    """
    if type(dataset) is list:
        dataset = np.asarray(dataset)
    result = list()
    for timestep in range(time_steps + 1):
        column = (np.roll(dataset, timestep, axis=0)).tolist()
        result.append(column)
    for nan_index in range(time_steps + 1):
        for index in range(nan_index):
            result[nan_index][index] = None

    result = list(reversed(result))

    # Transform rows to columns
    columns = list()
    for el_index in range(len(result[0])):
        tmp = list()
        for list_index in range(len(result)):
            tmp.append(result[list_index][el_index])
        columns.append(tmp)

    if dropnan:
        cleaned = []
        for sample in columns:
            if not None in sample:
                cleaned.append(sample)
        columns = cleaned

    # Split input and output values
    data_x = list()
    data_y = list()
    for sample in columns:
        tmp = sample
        data_y.append(tmp[-1])
        tmp.pop()
        data_x.append(tmp)

    return data_x, data_y
