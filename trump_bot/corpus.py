import json
import os
import string
from torch import zeros
from torch.autograd import Variable
from torch.tensor import Tensor
from tweet import decode_tweet, tweet
from unidecode import unidecode


class dictionary():
    '''
    A dictionary which contains all words in the training set.
    '''

    def __init__(self) -> None:
        '''
        Initialize the dictionary.
        '''

        self.idx2word: list[str] = []
        self.word2idx: dict[str, int] = {}

    def len(self) -> int:
        '''
        Return the current size of the dictionary.
        '''

        return len(self.idx2word)

    def add_word(self, word: str) -> int:
        '''
        Add a new word to the dictionary.

        Return the index of the word.

        :param word: new word
        '''

        if word not in self.idx2word:
            self.word2idx[word] = self.len()
            self.idx2word.append(word)
        return self.word2idx[word]

    def str2tensor(self, string: str) -> Variable:
        '''
        Convert a string to a list of tensors.

        Return a list of tensors.

        :param string: input string
        '''

        words: list[str] = string.split()
        tensor: Tensor = zeros(len(words)).long()
        for i in range(len(words)):
            tensor[i] = self.word2idx[words[i]]
        return Variable(tensor)


class corpus(dict):
    '''
    A corpus built with the training set.
    '''

    def __init__(self) -> None:
        '''
        Initialize the corpus.
        '''

        self.json_dir: str = os.path.realpath('data/raw_json')
        self.text_dir: str = os.path.realpath('data/text')
        self.default_src = 'data.txt'
        self.dictionary = dictionary()
        self.train_set: list[list[str]] = []

    def get_text_data(self, file_name: str, all_in_one: bool = False) -> None:
        '''
        Parse a dataset from JSON to plain text.

        :param file_name: file name of the dataset without extension
        :param all_in_one: write to a single file
        '''

        json_path: str = os.path.join(self.json_dir, file_name + '.json')
        try:
            with open(json_path, 'r', encoding='utf-8') as fi:
                data: list[dict] = json.load(fi)
        except FileNotFoundError:
            data: list[dict] = []

        text_name: str = self.default_src if all_in_one else file_name + '.txt'
        text_path: str = os.path.join(self.text_dir, text_name)
        with open(text_path, 'a' if all_in_one else 'w') as fo:
            for entry in data:
                t: tweet = decode_tweet(entry)
                fo.write(unidecode(t.text) + '\n')

    def get_all_text_data(self, all_in_one: bool = False) -> None:
        '''
        Parse all datasets in `json_dir` from JSON to plain text.

        :param all_in_one: write to a single file
        '''

        if all_in_one:
            # Clear the content
            text_path: str = os.path.join(self.text_dir, self.default_src)
            open(text_path, 'w').close()

        for json_entry in os.scandir(self.json_dir):
            file_name: str = json_entry.name
            if file_name.endswith('.json'):
                self.get_text_data(file_name[:-len('.json')], all_in_one)

    def read_data(self, file_name: str = None) -> None:
        '''
        Read a dataset from a file, and append to the corpus.

        :param file_name: file name of the dataset without extension
        '''

        text_name: str = file_name + '.txt' if file_name else self.default_src
        text_path: str = os.path.join(self.text_dir, text_name)
        with open(text_path, 'r') as fi:
            for line in fi:
                words: list[str] = line.split()
                if words[0].startswith('...'):
                    words.pop(0)
                if words[-1].endswith('...'):
                    words.pop(-1)
                else:
                    words.append('<eos>')
                self.train_set.append(words)
                for word in words:
                    self.dictionary.add_word(word)
