from corpus import corpus
from model import rnn
import os
from time import time
import torch
from torch import nn, optim
from torch.tensor import Tensor
from typing import List, Tuple
from utils import duration_since, plot


def init_corpus() -> None:
    '''
    Initialize a corpus. Read datasets from JSON files.
    '''

    global cp
    cp = corpus()
    # cp.get_all_text_data(all_in_one=False)
    # cp.read_data('2018')
    # cp.read_data('2019')
    cp.read_data('2020')
    cp.read_data('2021')
    print(f'Dictionary size: {cp.dictionary.len()}')


def init_model() -> None:
    '''
    Initialize the training model.
    '''

    dict_size = cp.dictionary.len()

    global m, criterion, optimizer
    m = rnn(dict_size, hidden_size, dict_size, num_layers, dropout).to(device)
    load_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(m.parameters(), learning_rate)


def words_to_tensor(words: List[str]) -> Tensor:
    '''
    Convert a sentence to a tensor.

    Return a tensor.

    :param words: a preprocessed word list of the sentence
    '''

    tensor: Tensor = torch.zeros(len(words), device=device).long()
    for i in range(len(words)):
        ran_i: int = torch.randint(
            cp.dictionary.start_pos, cp.dictionary.len(), (1,),
        )[0]
        tensor[i] = cp.dictionary.word2idx.get(words[i], ran_i)
    return tensor


def get_random_word() -> str:
    '''
    Return a random word from the dictionary.
    '''

    i: int = torch.randint(
        cp.dictionary.start_pos, cp.dictionary.len(), (1,),
    )[0]
    return cp.dictionary.idx2word[i]


def get_random_pair(dataset: str = 'train') -> Tuple[Tensor, Tensor]:
    '''
    Return a random pair of input and target from the dataset.

    :param dataset: which dataset, can be `'train'`, `'dev'` or `'test'`
    '''

    if dataset == 'dev':
        src = cp.dev_set
    elif dataset == 'test':
        src = cp.test_set
    else:
        src = cp.train_set

    max_i: int = len(src) - chunk_size
    i: int = torch.randint(0, max_i, (1,))[0]

    inp_words: List[str] = src[i:i+chunk_size]
    inp: Tensor = words_to_tensor(inp_words)

    tar_words: List[str] = src[i+1:i+1+chunk_size]
    tar: Tensor = words_to_tensor(tar_words)

    return inp, tar


def train(inp: Tensor, tar: Tensor) -> float:
    '''
    Train the model using a pair of input and target.

    Return the loss.

    :param inp: input tensor
    :param tar: target tensor
    '''

    m.zero_grad()
    hid: Tensor = m.init_hidden()
    loss: Tensor = 0

    for i in range(inp.size(0)):
        out, hid = m(inp[i], hid)
        loss += criterion(out, tar[i].view(-1))

    loss.backward()
    nn.utils.clip_grad_norm_(m.parameters(), clip)
    optimizer.step()

    return loss.item() / chunk_size


def train_model() -> List[float]:
    '''
    The main training function.

    Return all losses.
    '''

    m.train()
    all_losses: List[float] = []
    total_loss: float = 0.0
    min_loss: float = 2.0

    for epoch in range(1, num_epochs + 1):
        loss: float = train(*get_random_pair('train'))
        total_loss += loss

        if loss < min_loss:
            save_model()
            print(duration_since(start_time) + f': Model saved, {loss:.3f}')
            min_loss = loss

        if epoch % print_every == 0:
            progress: float = epoch / num_epochs * 100
            print('{}: ({} {:.1f}%) {:.3f}'.format(
                duration_since(start_time), epoch, progress, loss,
            ))
            evaluate_model()
            m.train()

        if epoch % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0.0

    return all_losses


def evaluate(prime_words: List[str] = None, predict_len: int = 30,
             temperature: float = 0.8) -> List[str]:
    '''
    Evaluate the network by generating a sentence using a priming word.

    To evaluate the network we feed one word at a time, use the outputs of the
    network as a probability distribution for the next word, and repeat.
    To start generation we pass some priming words to start setting up the
    hidden state, from which we then generate one word at a time.

    Return the predicted words.

    :param prime_words: priming words to start
    :param predict_len: expected length of words to predict
    :param temperature: randomness of predictions; higher value results in more diversity
    '''

    hid: Tensor = m.init_hidden()

    if not prime_words:
        prime_words = [cp.dictionary.sos]

    with torch.no_grad():
        prime_inp: Tensor = words_to_tensor(prime_words)
        predicted_words: List[str] = prime_words

        for p in range(len(prime_words) - 1):
            _, hid = m(prime_inp[p], hid)
        inp: Tensor = prime_inp[-1]

        for p in range(predict_len):
            out, hid = m(inp, hid)

            # Sample from the network as a multinomial distribution
            out_dist: Tensor = out.view(-1).div(temperature).exp()
            top_i: int = torch.multinomial(out_dist, 1)[0]

            # Add predicted word to words and use as next input
            predicted_word: str = cp.dictionary.idx2word[top_i]
            predicted_words.append(predicted_word)
            if (predicted_word == cp.dictionary.eos):
                break

            inp.fill_(top_i)

    return predicted_words


def evaluate_model() -> None:
    '''
    The main evaluating function.
    '''

    m.eval()
    prime_word: str = get_random_word()
    predicted_words: List[str] = evaluate(
        [cp.dictionary.sos, prime_word], predict_len, temperature,
    )
    print(' '.join(predicted_words))


def save_model() -> None:
    '''
    Save the current model.
    '''

    with open(save_path, 'wb') as f:
        torch.save(m.state_dict(), f)


def load_model() -> None:
    '''
    Load the best model from file.
    '''

    try:
        with open(save_path, 'rb') as f:
            m.load_state_dict(torch.load(f))
    except FileNotFoundError:
        pass


def main() -> None:
    '''
    The main function of Trump-bot.
    '''

    init_corpus()
    print(duration_since(start_time) + ': Reading dataset done.')

    init_model()
    print(duration_since(start_time) + ': Training model initialized.')

    all_losses = train_model()
    print(duration_since(start_time) + ': Training model done.')

    plot(all_losses)
    print(duration_since(start_time) + ': Plotting done.')

    load_model()
    for i in range(1, batch_size + 1):
        progress: float = i / batch_size * 100
        print(f'({i} {progress:.1f}%)')
        evaluate_model()
    print(duration_since(start_time) + ': Evaluating model done.')


if __name__ == '__main__':
    # Parameters
    hidden_size = 1000
    num_layers = 3
    dropout = 0.2
    learning_rate = 0.001
    num_epochs = 4000
    batch_size = 30
    chunk_size = 30
    predict_len = 140
    temperature = 0.7
    clip = 0.25
    random_seed = 1234
    print_every = 100
    plot_every = 10
    save_path = os.path.realpath('model/model.pt')

    # Set the random seed manually for reproducibility.
    torch.manual_seed(random_seed)
    # Enable CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start_time = time()

    try:
        main()
    except KeyboardInterrupt:
        print('\nAborted.')
