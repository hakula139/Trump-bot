import torch
from torch import nn
from torch.tensor import Tensor
from typing import Tuple


class rnn(nn.Module):
    '''
    Build an RNN model.

    This model will take the last character as input and is expected to output
    the next character. There are three layers - one linear layer that encodes
    the input character into an internal state, one GRU layer (which may itself
    have multiple layers) that operates on that internal state and a hidden
    state, and a decoder layer that outputs the probability distribution.
    '''

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int = 1, dropout: float = 0.2) -> None:
        '''
        Initialize the RNN model.

        :param input_size: the number of expected features in the input
        :param hidden_size: the number of features in the hidden state
        :param output_size: the number of expected features in the output
        :param num_layers: the number of recurrent layers
        '''

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(output_size, input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, inp: Tensor, hid: Tensor) -> Tuple[Tensor, Tensor]:
        '''
        The forward function which defines the network structure.

        Return the result of output tensor and hidden tensor.

        :param inp: input tensor
        :param hid: hidden tensor
        '''

        emb: Tensor = self.drop(self.encoder(inp.view(1, -1)))
        out, hid = self.gru(emb, hid)
        out: Tensor = self.drop(out)
        dec: Tensor = self.decoder(out.view(1, -1))
        return dec, hid

    def init_hidden(self) -> Tensor:
        '''
        Initialize the hidden state.
        '''

        weight = next(self.parameters())
        return weight.new_zeros(self.num_layers, 1, self.hidden_size)
