import numpy as np


def frame_data(dataset, look_back=1):
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

    return columns


x = frame_data(dataset=[1, 2, 3, 4, 5, 6, 7], look_back=1)
print(x)

x = frame_data(dataset=[[1,0],[0,1],[1,1], [1,1], [1,1]], look_back=1)
print(x)

