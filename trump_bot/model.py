from torch import nn, zeros
from torch.autograd import Variable
from torch.tensor import Tensor
from typing import Tuple


class RNN(nn.Module):
    '''
    Build an RNN model.

    This model will take the last character as input and is expected to output
    the next character. There are three layers - one linear layer that encodes
    the input character into an internal state, one GRU layer (which may itself
    have multiple layers) that operates on that internal state and a hidden
    state, and a decoder layer that outputs the probability distribution.
    '''

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int = 1) -> None:
        '''
        Initialize the RNN model.

        :param input_size: the number of expected features in the input
        :param hidden_size: the number of features in the hidden state
        :param output_size: the number of features in the output
        :param num_layers: the number of recurrent layers
        '''

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        '''
        The forward function which defines the network structure.

        Return the result of output tensor and hidden tensor.

        :param input: input tensor
        :param hidden: hidden tensor
        '''

        input: Tensor = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output: Tensor = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self) -> Variable:
        '''
        Initialize the hidden state.
        '''

        return Variable(zeros(self.num_layers, 1, self.hidden_size))
