"""
      _             _                              _                   
     | |           | |                            | |                  
  ___| |_ __ _  ___| | __  ___ _ __ ___   __ _ ___| |__   ___ _ __ ___ 
 / __| __/ _` |/ __| |/ / / __| '_ ` _ \ / _` / __| '_ \ / _ \ '__/ __|
 \__ \ || (_| | (__|   <  \__ \ | | | | | (_| \__ \ | | |  __/ |  \__ \
 |___/\__\__,_|\___|_|\_\ |___/_| |_| |_|\__,_|___/_| |_|\___|_|  |___/

@author: Andrea Simeoni 21 ott 2017   
https://github.com/insanediv/machine-learning-tuts/blob/master/neural_nets/hello_lstm.py
"""
from time import sleep

import tflearn

from neural_nets.data_helper import DataHelper
from neural_nets.framing import frame_data

file_path = 'raw_data/hello.txt'
data_helper = DataHelper(file_path=file_path)
one_hot_dict = data_helper.one_hot_dictionary

# model params
n_epoch = 1000
time_steps = 3
hidden_state_size = 128
batch_size = 20
sample_length = data_helper.get_input_vector_size()

# No framed data (chars sequence one-hot encoded)
one_hot_train_data = data_helper.get_next_batch(batch_size)
# Framed data one-hot encoded
trainX, trainY = frame_data(dataset=one_hot_train_data, time_steps=time_steps)

# Recurrent Neural Network model
net = tflearn.input_data(shape=[None, time_steps, sample_length], name='input')
net = tflearn.lstm(net, hidden_state_size, dropout=0.8)
net = tflearn.fully_connected(net, sample_length, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, show_metric=True, n_epoch=n_epoch, batch_size=batch_size)

# Generate output sequences
test_data = data_helper.interactive_input(input_len=time_steps)
prediction = model.predict([test_data])
pred_char, pred_one_hot = data_helper.get_prediction(prediction)
print(pred_char)
while True:
    test_data = test_data[1:]
    test_data.append(pred_one_hot)
    prediction = model.predict([test_data])
    pred_char, pred_one_hot = data_helper.get_prediction(prediction)
    print(pred_char, end=' ', flush=True)
    sleep(0.7)
