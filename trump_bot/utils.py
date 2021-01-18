import matplotlib.pyplot as plt
from math import floor
import os
from time import time
from typing import List

plot_location: str = os.path.realpath('assets/loss.png')


def duration_since(start_time: float) -> str:
    '''
    Return the duration since start time in a human-readable format.

    :param start_time: start time
    '''

    duration = time() - start_time
    hour = floor(duration / 3600.0)
    duration -= hour * 3600.0
    minute = floor(duration / 60.0)
    duration -= minute * 60.0
    second = duration

    pretty_duration = ''
    if hour:
        pretty_duration += f'{hour}h '
    if minute:
        pretty_duration += f'{minute}m '
    pretty_duration += f'{second:.1f}s'
    return pretty_duration


def plot(
    num_epochs: int, plot_every: int, all_train_losses: List[float], all_valid_losses: List[float]
) -> None:
    '''
    Plot the historical training loss and validation loss, so as to show the
    network learning rate.

    :param num_epochs: the number of epochs
    :param plot_every: plotting interval
    :param all_train_losses: the historical training loss
    :param all_valid_losses: the historical validation loss
    '''

    epochs = range(1, num_epochs + 1, plot_every)
    plt.figure()
    plt.plot(epochs, all_train_losses, 'r', label='Training loss')
    plt.plot(epochs, all_valid_losses, 'b', label='Validation loss')
    plt.title('Learning rate')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(plot_location)
