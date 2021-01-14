import json
import os
import string
from tweet import decode_tweet, tweet
from unidecode import unidecode


class corpus(dict):
    '''
    Corpus built with our training dataset.
    '''

    def __init__(self) -> None:
        '''
        Initialize corpus.
        '''

        self.json_dir: str = os.path.realpath('data/raw_json')
        self.text_dir: str = os.path.realpath('data/text')
        self.characters: str = string.printable

    def get_text_data(self, file_name: str, all_in_one: bool = False) -> None:
        '''
        Parse dataset from JSON to plain text.

        :param file_name: file name of the dataset without extension
        :param all_in_one: write to a single file
        '''

        json_path: str = os.path.join(self.json_dir, file_name + '.json')
        try:
            with open(json_path, 'r', encoding='utf-8') as fi:
                data: list[dict] = json.load(fi)
        except FileNotFoundError:
            data: list[dict] = []

        text_name = 'data' if all_in_one else file_name
        text_path: str = os.path.join(self.text_dir, text_name + '.txt')
        with open(text_path, 'a' if all_in_one else 'w') as fo:
            for entry in data:
                t: tweet = decode_tweet(entry)
                fo.write(unidecode(t.text) + '\n')

    def get_all_text_data(self, all_in_one: bool = False) -> None:
        '''
        Parse all dataset in `json_dir` from JSON to plain text.

        :param all_in_one: write to a single file
        '''

        if all_in_one:
            # Clear the content
            text_path: str = os.path.join(self.text_dir, 'data.txt')
            open(text_path, 'w').close()

        for json_entry in os.scandir(self.json_dir):
            file_name = json_entry.name
            if file_name.endswith('.json'):
                self.get_text_data(file_name[:-len('.json')], all_in_one)
