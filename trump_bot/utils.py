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


def plot(all_losses: List[float]) -> None:
    '''
    Plot the historical loss from `all_losses`, which shows the network learning rate.

    :param all_losses: the historical loss
    '''

    plt.figure()
    plt.plot(all_losses)
    plt.savefig(plot_location)
