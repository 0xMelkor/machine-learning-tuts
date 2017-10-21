"""
      _             _                              _                   
     | |           | |                            | |                  
  ___| |_ __ _  ___| | __  ___ _ __ ___   __ _ ___| |__   ___ _ __ ___ 
 / __| __/ _` |/ __| |/ / / __| '_ ` _ \ / _` / __| '_ \ / _ \ '__/ __|
 \__ \ || (_| | (__|   <  \__ \ | | | | | (_| \__ \ | | |  __/ |  \__ \
 |___/\__\__,_|\___|_|\_\ |___/_| |_| |_|\__,_|___/_| |_|\___|_|  |___/

@author: Andrea Simeoni 21 ott 2017   
https://github.com/insanediv/machine-learning-tuts/blob/master/neural_nets/data_helper.py
"""
import numpy as np


class DataHelper:
    def __init__(self, file_path):
        self.file_path = file_path
        self.next_batch = 1
        self.dictionary = self._extract_char_dictionary()
        self.one_hot_dictionary = self.get_one_hot_dictionary()
        self.one_hot_data = self._load_onehot_data()

    def get_next_batch(self, batch_size=1):
        start_index = batch_size * self.next_batch
        end_index = start_index + batch_size
        result = self.one_hot_data[start_index:end_index]
        if len(result) is 0:
            self.next_batch = 1
            result = self.one_hot_data[batch_size * self.next_batch:batch_size]
        else:
            self.next_batch += 1
        return list(result)

    def get_input_vector_size(self):
        return len(self.dictionary)

    def get_one_hot_dictionary(self):
        """
        :return: Dictionary of one-hot associations {'level-character': [0,1,0,0,0]}
        """
        categories = np.array(self.dictionary)
        vect_length = len(categories)
        one_hot_vectors = np.zeros(shape=[vect_length, vect_length])

        one_hot_vectors[np.arange(vect_length), np.arange(vect_length)] = 1

        dictionary = dict()
        for index in range(vect_length):
            dictionary[categories[index]] = list(one_hot_vectors[index])

        return dictionary

    def _load_onehot_data(self):
        """
        :return: nparray of one-hot encoded characters from the whole set of characters in the dataset
        """
        result = list()
        with open(self.file_path, 'r') as f:
            for line in f:
                line_chars = list(line)
                for character in line_chars:
                    if character != '\n':
                        result.append(self.one_hot_dictionary[character])

        return result

    def _extract_char_dictionary(self):
        """
        :return: List of characters in the data file
        """
        level_charset = set()
        with open(self.file_path, 'r') as f:
            for line in f:
                level_charset |= set(line)
        level_charset.remove('\n')
        return list(level_charset)

    def get_value_from_one_hot(self, one_hot):
        val = one_hot.index(max(one_hot))
        for key in self.one_hot_dictionary:
            max_value = self.one_hot_dictionary[key].index(max(self.one_hot_dictionary[key]))
            if val == max_value:
                return key
        return None

    def get_values_from_one_hot_list(self, one_hot_values):
        if type(one_hot_values) is np.ndarray:
            one_hot_values = one_hot_values.tolist()
        result = list()
        for one_hot in one_hot_values:
            result.append(self.get_value_from_one_hot(one_hot))

        return result

    def get_prediction(self, pred):
        predicted_char = self.get_values_from_one_hot_list(pred)[0]
        predicted_one_hot = self.one_hot_dictionary[predicted_char]
        return predicted_char, predicted_one_hot

    def interactive_input(self, input_len):
        user_msg = 'Insert %d characters among %s \n' % (input_len, self.dictionary)

        result = list()
        while len(result) != input_len:
            user_input = input(user_msg)
            char_list = list(user_input)
            if len(char_list) == input_len:
                one_hot_dict = self.get_one_hot_dictionary()
                for characther in char_list:
                    try:
                        result.append(one_hot_dict[characther])
                    except KeyError:
                        continue
        return result
