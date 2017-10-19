from neural_nets.data_helper import DataHelper

file_path = 'raw_data/hello.txt'
data_helper = DataHelper(file_path=file_path)

one_hot_dictionaries = data_helper.get_one_hot_dictionaries()
print(one_hot_dictionaries)