import torch
import pickle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self



class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    #def __init__(self, character, separator = []):
    def __init__(self, character, separator_list = {}, dict_pathlist = {}):
        # character (str): set of the possible characters.
        dict_character = list(character)

        #special_character = ['\xa2', '\xa3', '\xa4','\xa5']
        #self.separator_char = special_character[:len(separator)]

        self.dict = {}
        #for i, char in enumerate(self.separator_char + dict_character):
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[blank]'] + dict_character  # dummy '[blank]' token for CTCLoss (index 0)
        #self.character = ['[blank]']+ self.separator_char + dict_character  # dummy '[blank]' token for CTCLoss (index 0)
        self.separator_list = separator_list

        separator_char = []
        for lang, sep in separator_list.items():
            separator_char += sep

        self.ignore_idx = [0] + [i+1 for i,item in enumerate(separator_char)]

        dict_list = {}
        for lang, dict_path in dict_pathlist.items():
            with open(dict_path, "rb") as input_file:
                word_count = pickle.load(input_file)
            dict_list[lang] = word_count
        self.dict_list = dict_list

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode_greedy(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] not in self.ignore_idx and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank (and separator).
                #if (t[i] != 0) and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank (and separator).
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts

    



