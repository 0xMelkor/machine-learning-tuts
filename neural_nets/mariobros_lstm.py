
from neural_nets.framing import series_to_supervised as frame_data
from neural_nets.data_helper import DataHelper

data_helper = DataHelper()
one_hot_dict = data_helper.get_one_hot_dictionaries()
input_dim = data_helper.get_input_vector_size()

# Network parameters
n_units = 128
dropout = 0.8
timesteps = 5
learning_rate = 0.001
batch_size = 100

sequence_data = data_helper.get_next_batch(batch_size=batch_size)
framed_data = frame_data(sequence_data, timesteps, 1)

print(framed_data)

# Network building
"""net = tflearn.input_data([None, input_dim])
net = tflearn.lstm(net, n_units, dropout)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate,
                         loss='categorical_crossentropy')
"""