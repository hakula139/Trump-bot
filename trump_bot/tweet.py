from datetime import datetime
from typing import NamedTuple


class tweet(NamedTuple):
    '''
    Data model for a tweet from Twitter.
    '''

    date: str
    favorites: int
    id: int
    is_retweet: bool
    retweets: int
    text: str


def decode_tweet(tweet_json: dict) -> tweet:
    '''
    Decode a tweet in JSON syntax to a named tuple.

    Return a tweet of `tweet` type.

    :param tweet_json: a tweet in JSON syntax
    '''

    tweet_date: str = (
        datetime.fromtimestamp(tweet_json['date'] / 1000.0)
        .strftime('%Y-%m-%d %H:%M:%S')
    )
    return tweet(
        tweet_date,
        tweet_json['favorites'],
        tweet_json['id'],
        tweet_json['isRetweet'],
        tweet_json['retweets'],
        tweet_json['text'],
    )
