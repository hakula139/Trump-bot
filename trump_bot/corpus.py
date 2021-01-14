import json
import os
from typing import Dict, List
from torch import zeros
from torch.autograd import Variable
from torch.tensor import Tensor
from torchtext.data import get_tokenizer
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

        self.idx2word: List[str] = []
        self.word2idx: Dict[str, int] = {}

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

        words: List[str] = string.split()
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
        self.train_set_file = 'train.txt'
        self.train_set: List[List[str]] = []
        self.dictionary = dictionary()

    def get_text_data(self, file_name: str, all_in_one: bool = False) -> None:
        '''
        Parse a dataset from JSON to plain text.

        :param file_name: file name of the dataset without extension
        :param all_in_one: write to a single file
        '''

        json_path: str = os.path.join(self.json_dir, file_name + '.json')
        try:
            with open(json_path, 'r', encoding='utf-8') as fi:
                data: List[dict] = json.load(fi)
        except FileNotFoundError:
            data: List[dict] = []

        text_name: str = self.train_set_file if all_in_one else file_name + '.txt'
        text_path: str = os.path.join(self.text_dir, text_name)
        buffer_size = 1 << 20  # 1 MB
        tokenizer = get_tokenizer('spacy')
        with open(text_path, 'a' if all_in_one else 'w', buffering=buffer_size) as fo:
            buffer: str = ''
            for entry in data:
                t: tweet = decode_tweet(entry)
                text: str = unidecode(t.text)
                words: List[str] = tokenizer(text)
                buffer += ' '.join(words) + '\n'
                self.add_sentence(words)
            fo.write(buffer)

    def get_all_text_data(self, all_in_one: bool = False) -> None:
        '''
        Parse all datasets in `json_dir` from JSON to plain text.

        :param all_in_one: write to a single file
        '''

        if all_in_one:
            # Clear the content
            text_path: str = os.path.join(self.text_dir, self.train_set_file)
            open(text_path, 'w').close()

        for json_entry in os.scandir(self.json_dir):
            file_name: str = json_entry.name
            if file_name.endswith('.json'):
                self.get_text_data(file_name[:-len('.json')], all_in_one)

    def add_sentence(self, words: List[str]) -> None:
        '''
        Add a new sentence to the corpus.

        :param words: a preprocessed word list of the new sentence
        '''

        if not words:
            return
        try:
            if words[0].startswith('...'):
                words.pop(0)
            else:
                words.append('<sos>')
            if words[-1].endswith('...'):
                words.pop(-1)
            else:
                words.append('<eos>')
        except IndexError:
            pass
        else:
            self.train_set.append(words)
            for word in words:
                self.dictionary.add_word(word)

    def read_data(self, file_name: str = None) -> None:
        '''
        Read a dataset from a file, and append to the corpus.

        :param file_name: file name of the dataset without extension
        '''

        text_name: str = file_name + '.txt' if file_name else self.train_set_file
        text_path: str = os.path.join(self.text_dir, text_name)
        with open(text_path, 'r') as fi:
            for line in fi:
                self.add_sentence(line.split())
