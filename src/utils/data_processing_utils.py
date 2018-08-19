import re
from typing import Tuple
import ast


def prepare_input(input_text: str) -> list:
    """Removes everything except letters, numbers and punctuation for a string.
       Additionally it converts the words to lowecase

    Args:
        input_text: A future input to the model

    Returns:
        words: A list of words

    """
    regex = re.compile("[^a-zA-Z0-9,\.!?']")
    raw_words = regex.sub(" ", input_text).lower().split()

    return raw_words


def replace_with_unk(input_words: list, freq: dict) -> list:
    """Given a list of words it will replace the words
       that have frequency lower than freq with <unk>

    Args:
        input_words: A list of words
        freq: Dictionary that contains that frequency of all words

    Returns:
        output_words: A list of words

    """
    output_words = [word if freq[word] > 5 else "<unk>" for word in input_words]

    return output_words


def prepare_movie_line(movie_line_raw: bytearray) -> Tuple:
    """Prepares raw movie line for processing

    Args:
        movie_line_raw: A raw movie line

    Returns:
        movie_line_clean: A cleaned movie line

    """
    line = movie_line_raw.decode("utf-8", errors="replace")
    movie_line_parts = line.split(" +++$+++ ")
    id = movie_line_parts[0]
    movie_line_clean = movie_line_parts[-1]

    return id, movie_line_clean


def prepare_movie_conv(movie_conv_raw: bytearray) -> list:
    """Prepares one movie conversation for processing

    Args:
        movie_conv_raw: A raw movie conversation

    Returns:
        movie_conv_clean: A cleaned movie conversation

    """
    line = movie_conv_raw.decode("utf-8", errors="replace")
    conversation = ast.literal_eval(line.split(" +++$+++ ")[-1])

    return conversation

def replace_with_unk_chat(input_words: list, word2index: dict) -> list:
    """

    Args:
        input_words: A list of words
        freq: Dictionary that contains that frequency of all words

    Returns:
        output_words: A list of words

    """
    output_words = [word if word in word2index else "<unk>" for word in input_words]

    return output_words