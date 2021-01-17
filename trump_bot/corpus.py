import json
from math import floor
import os
from typing import Dict, List
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
        self.idx2freq: List[int] = []
        self.word2idx: Dict[str, int] = {}
        self.freq_threshold = 1
        self.start_pos = 3

        self.unk = '<unk>'  # unknown word
        self.eos = '<eos>'  # end of sentence
        self.sos = '<sos>'  # start of sentence

        self.add_word(self.unk)
        self.add_word(self.eos)
        self.add_word(self.sos)

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
            idx = self.word2idx[word] = self.len()
            self.idx2word.append(word)
            self.idx2freq.append(1)
        else:
            idx = self.word2idx[word]
            self.idx2freq[idx] += 1
        return idx

    def clear_words(self) -> None:
        '''
        Remove infrequent words that appears at most `freq_threshold`.
        '''

        i: int = self.start_pos
        while i < len(self.idx2word):
            if self.idx2freq[i] and self.idx2freq[i] <= self.freq_threshold:
                removed_word: str = self.idx2word[i]
                self.word2idx.pop(removed_word)

                last_word: str = self.idx2word.pop()
                if i >= len(self.idx2word):
                    self.idx2freq.pop()
                    break

                # Swap the removed word with the last word in dictionary
                self.idx2word[i] = last_word
                self.idx2freq[i] = self.idx2freq.pop()
                self.word2idx[last_word] = i
            else:
                i += 1


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
        self.data_file = 'data.txt'

        self.train_set: List[str] = []
        self.train_proportion = 0.8

        self.dev_set: List[str] = []
        self.dev_proportion = 0.1

        self.test_set: List[str] = []
        self.test_proportion = 0.1

        self.dictionary = dictionary()

    def get_text_data(self, file_name: str, all_in_one: bool = False) -> None:
        '''
        Parse a dataset from JSON to plain text.

        :param file_name: file name of the dataset without extension
        :param all_in_one: write to a single file
        '''

        def _filter_text(text: str) -> str:
            '''
            Filter a line of text and replace certain words.

            Return the filtered text.

            :param text: input text
            '''

            return (
                text
                .replace('&amp;', '&')
                .replace('&amp,', '&')
            )

        json_path: str = os.path.join(self.json_dir, file_name + '.json')
        try:
            with open(json_path, 'r', encoding='utf-8') as fi:
                data: List[dict] = json.load(fi)
        except FileNotFoundError:
            data: List[dict] = []

        text_name: str = self.data_file if all_in_one else file_name + '.txt'
        text_path: str = os.path.join(self.text_dir, text_name)
        buffer_size = 1 << 20  # 1 MB
        tokenizer = get_tokenizer('spacy')
        with open(text_path, 'a' if all_in_one else 'w', buffering=buffer_size) as fo:
            buffer: str = ''
            # Reverse the list to sort by time in ascending order
            for entry in reversed(data):
                t: tweet = decode_tweet(entry)
                text: str = _filter_text(unidecode(t.text))
                words: List[str] = tokenizer(text)
                buffer += ' '.join(words) + '\n'
            fo.write(buffer)

    def get_all_text_data(self, all_in_one: bool = False) -> None:
        '''
        Parse all datasets in `json_dir` from JSON to plain text.

        :param all_in_one: write to a single file
        '''

        if all_in_one:
            # Clear the content
            text_path: str = os.path.join(self.text_dir, self.data_file)
            open(text_path, 'w').close()

        for json_entry in os.scandir(self.json_dir):
            file_name: str = json_entry.name
            if file_name.endswith('.json'):
                self.get_text_data(file_name[:-len('.json')], all_in_one)

    def add_sentence(self, words: List[str], dataset: str = 'train') -> None:
        '''
        Add a new sentence to the corpus.

        :param words: a preprocessed word list of the new sentence
        :param dataset: which dataset, can be `'train'`, `'dev'` or `'test'`
        '''

        if not words:
            return
        try:
            if words[0].startswith('...'):
                words.pop(0)
            else:
                words.insert(0, self.dictionary.sos)
            if words[-1].endswith('...'):
                words.pop(-1)
            else:
                words.append(self.dictionary.eos)
        except IndexError:
            pass
        else:
            for word in words:
                self.dictionary.add_word(word)
            if dataset == 'dev':
                self.dev_set += words
            elif dataset == 'test':
                self.test_set += words
            else:
                self.train_set += words

    def read_data(self, file_name: str = None) -> None:
        '''
        Read a dataset from a file, and append to the corpus.

        :param file_name: file name of the dataset without extension
        '''

        text_name: str = file_name + '.txt' if file_name else self.data_file
        text_path: str = os.path.join(self.text_dir, text_name)

        with open(text_path, 'r') as fi:
            all_lines: List[str] = fi.read().splitlines()
            train_size: int = floor(len(all_lines) * self.train_proportion)
            dev_size: int = floor(len(all_lines) * self.dev_proportion)

            for line in all_lines[:train_size]:
                self.add_sentence(line.split(), 'train')
            for line in all_lines[train_size:train_size+dev_size]:
                self.add_sentence(line.split(), 'dev')
            for line in all_lines[train_size+dev_size:]:
                self.add_sentence(line.split(), 'test')

            self.dictionary.clear_words()
