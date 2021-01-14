from corpus import corpus
import time
from torch.autograd import Variable
from typing import List, Tuple


def main() -> None:
    '''
    Main function for bot.
    '''

    current_time = time.time()

    c = corpus()
    c.get_all_text_data(all_in_one=True)
    d = c.dictionary

    duration = time.time() - current_time
    print(f'Reading dataset done: {duration:.2f} s')
    current_time += duration

    train_pairs: List[Tuple[Variable, Variable]] = []
    for sentence in c.train_set:
        input: Variable = d.words2tensor(sentence[:-1])
        target: Variable = d.words2tensor(sentence[1:])
        train_pairs.append((input, target))

    duration = time.time() - current_time
    print(f'Training set established: {duration:.2f} s')
    current_time += duration


if __name__ == '__main__':
    main()
