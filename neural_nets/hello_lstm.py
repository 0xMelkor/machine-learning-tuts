import tflearn
from time import sleep
from neural_nets.data_helper import DataHelper
from neural_nets.framing import frame_data

file_path = 'raw_data/hello.txt'
data_helper = DataHelper(file_path=file_path)
one_hot_dict = data_helper.one_hot_dictionary;

# model params
time_steps = 3
sample_length = data_helper.get_input_vector_size()

one_hot_train_data = data_helper.get_next_batch(batch_size=20)
one_hot_test_data = data_helper.get_next_batch(batch_size=20)

trainX, trainY = frame_data(dataset=one_hot_train_data, look_back=time_steps)

net = tflearn.input_data(shape=[None, time_steps, sample_length],name='input')
net = tflearn.lstm(net, 50, dropout=0.8)
net = tflearn.fully_connected(net, sample_length, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, show_metric=True, n_epoch=400, batch_size=20)

test_data = data_helper.interactive_input(input_len=time_steps)
prediction = model.predict([test_data])
pred_char, pred_one_hot = data_helper.get_prediction(prediction)
print(pred_char)
while True:
    test_data = test_data[1:]
    test_data.append(pred_one_hot)
    prediction = model.predict([test_data])
    pred_char, pred_one_hot = data_helper.get_prediction(prediction)
    print(pred_char)
    sleep(0.2)

