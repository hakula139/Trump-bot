from corpus import corpus
from math import floor
from model import rnn
from time import time
import torch
from torch import nn
from torch.tensor import Tensor
from typing import List, Tuple

# Parameters
hidden_size = 100
num_layers = 4
num_epochs = 2000
chunk_size = 30
random_seed = 1234
print_every = 100
plot_every = 10

# Set the random seed manually for reproducibility.
torch.manual_seed(random_seed)
# Enable CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def duration_since(start_time: float) -> str:
    '''
    Return the duration since start time in a human-readable format.

    :param start_time: start time
    '''

    duration: float = time() - start_time
    minute: int = floor(duration / 60.0)
    second: float = duration - minute * 60.0
    return (f'{minute} min ' if minute else '') + f'{second:.2f} sec'


def init_corpus() -> None:
    '''
    Initialize a corpus. Read datasets from JSON files.
    '''

    global cp
    cp = corpus()
    # cp.get_all_text_data(all_in_one=True)
    cp.read_data()


def init_model() -> None:
    '''
    Initialize the training model.
    '''

    dict_size = cp.dictionary.len()

    global m, criterion
    m = rnn(dict_size, hidden_size, dict_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()


def get_train_pair() -> Tuple[Tensor, Tensor]:
    '''
    Return a random pair of input and target based on the source tensor.

    :param src: source tensor
    '''

    max_i: int = len(cp.train_set) - chunk_size
    # Take a random integer from [0, max_i)
    i: int = torch.randint(0, max_i, (1,))[0]
    inp_words: List[str] = cp.train_set[i:i+chunk_size]
    tar_words: List[str] = cp.train_set[i+1:i+1+chunk_size]
    inp: Tensor = cp.dictionary.words2tensor(inp_words).to(device)
    tar: Tensor = cp.dictionary.words2tensor(tar_words).to(device)
    return (inp, tar)


def train(inp: Tensor, tar: Tensor) -> float:
    '''
    The main training function.

    Return the total loss.

    :param inp: input tensor
    :param tar: target tensor
    '''

    m.train()
    hid: Tensor = m.init_hidden().to(device)
    m.zero_grad()
    total_loss: Tensor = torch.tensor([0.0], requires_grad=True)

    for i in range(chunk_size):
        out, hid = m.forward(inp[i], hid)
        total_loss += criterion(out, tar[i])

    total_loss.backward()
    return total_loss[0] / chunk_size


def evaluate(prime_words: List[str] = ['<sos>'], predict_len: int = 30,
             temperature: float = 0.8) -> List[str]:
    '''
    Evaluate the network.

    To evaluate the network we feed one word at a time, use the outputs of the
    network as a probability distribution for the next word, and repeat.
    To start generation we pass some priming words to start setting up the
    hidden state, from which we then generate one word at a time.

    Return the predicted words.

    :param prime_words: priming words to start
    :param predict_len: expected length of words to predict
    :param temperature: randomness of predictions; higher value results in more diversity
    '''

    m.eval()
    hid: Tensor = m.init_hidden()
    prime_inp: Tensor = cp.dictionary.words2tensor(prime_words)
    predicted_words: List[str] = prime_words

    for p in range(len(prime_words) - 1):
        _, hid = m.forward(prime_inp[p], hid)
    inp: Tensor = prime_inp[-1]

    for p in range(predict_len):
        out, hid = m.forward(inp, hid)

        # Sample from the network as a multinomial distribution
        out_dist: Tensor = out.view(-1).div(temperature).exp()
        top_i: int = torch.multinomial(out_dist, 1)[0]

        # Add predicted word to words and use as next input
        predicted_word: str = cp.dictionary.idx2word[top_i]
        predicted_words.append(predicted_word)
        inp = cp.dictionary.words2tensor(predicted_word)

    return predicted_words


def main() -> None:
    '''
    Main function for bot.
    '''

    start_time = time()

    init_corpus()
    print(duration_since(start_time) + ': Reading dataset done.')

    init_model()
    print(duration_since(start_time) + ': Training model initialized.')

    all_losses: List[float] = []
    loss_avg: float = 0.0

    for epoch in range(1, num_epochs + 1):
        loss: float = train(*get_train_pair())
        loss_avg += loss

        if epoch % print_every == 0:
            progress: float = epoch / num_epochs * 100
            print()
            print('{:4} ({:5}%%) [{}] - Loss: {}'.format(
                epoch, progress, duration_since(start_time), loss)
            )
            print(evaluate('<sos>', 100))

        if epoch % plot_every == 0:
            all_losses.append(loss_avg / plot_every)
            loss_avg = 0.0


if __name__ == '__main__':
    main()
