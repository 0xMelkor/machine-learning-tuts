import numpy as np


def frame_data(dataset, look_back=1, dropnan=True):
    """
    Data framing inspired to
    https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
    :param dataset: set of sequenced data
    :param look_back: how many timesteps to look backward
    :param dropnan: because of shifting some rows have None objects. Set true to remove such rows
    :return: 
    """
    if type(dataset) is list:
        dataset = np.asarray(dataset)
    result = list()
    for timestep in range(look_back + 1):
        column = (np.roll(dataset, timestep, axis=0)).tolist()
        result.append(column)
    for nan_index in range(look_back + 1):
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
    dataX = list()
    dataY = list()
    for sample in columns:
        tmp = sample
        dataY.append(tmp[-1])
        tmp.pop()
        dataX.append(tmp)

    return dataX, dataY
