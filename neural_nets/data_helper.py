import numpy as np


class DataHelper:
    def __init__(self, file_path):
        self.file_path = file_path
        self.next_batch = 1
        self.dictionary = self._extract_dictionary()
        self.one_hot_data = self._load_onehot_data()

    def get_next_batch(self, batch_size=1):
        start_index = batch_size*self.next_batch
        end_index = start_index + batch_size
        result = self.one_hot_data[start_index:end_index]
        if len(result) is 0:
            self.next_batch = 1
            result = self.one_hot_data[batch_size*self.next_batch:batch_size]
        else:
            self.next_batch += 1
        print(self.next_batch)
        return list(result)

    def get_input_vector_size(self):
        return len(self.dictionary)

    def get_one_hot_dictionaries(self):
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
        data_dictionary = self.get_one_hot_dictionaries()
        result = list()
        with open(self.file_path, 'r') as f:
            for line in f:
                line_chars = list(line)
                for character in line_chars:
                    if character != '\n':
                        result.append(data_dictionary[character])

        return result

    def _extract_dictionary(self):
        """
        :return: List of characters in the data file
        """
        level_charset = set()
        with open(self.file_path, 'r') as f:
            for line in f:
                level_charset |= set(line)
        level_charset.remove('\n')
        return list(level_charset)