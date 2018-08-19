import os
import pickle
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


def load_vocabs_pickles(
    pickle_dir_path: str, word2index_b: bool, index2word_b: bool
) -> (Dict, Dict):
    """Loads the needed pickles

    Args:
        pickle_dir_path: A path to the pickle dir
        word2index: Whether to load word2index
        index2word: Whether to load index2word

    Returns:
        word2index: The word2index dict
        index2word: The index2word dict

    """
    pickle_word2index = os.path.join(pickle_dir_path, "word2index.pkl")
    pickle_index2word = os.path.join(pickle_dir_path, "index2word.pkl")
    if word2index_b and index2word_b:
        word2index = pickle.load(open(pickle_word2index, "rb"))
        index2word = pickle.load(open(pickle_index2word, "rb"))
        return word2index, index2word
    elif word2index_b:
        word2index = pickle.load(open(pickle_word2index, "rb"))
        return word2index
    elif index2word_b:
        index2word = pickle.load(open(pickle_index2word, "rb"))
        return index2word
    else:
        log.info("One of word2index and index2word must be true to load some pikcle")


def load_input_output_lengths_pickles(
    pickle_dir_path: str, io_b: bool, iol_b: bool
) -> Tuple:
    """Loads the inputs, outputs and length pickles

    Args:
        pickle_dir_path: A path to the pickle dir
        io_b: Whether to load inputs and outputs pickles
        iol_b: Whether to load inputs and outputs lengths pickles

    Returns:
        inputs: Inputs list
        outputs: Outputs list
        inputs_lengths: I-lengths list
        outputs_length: O-lengths list

    """
    pickle_inputs = os.path.join(pickle_dir_path, "inputs.pkl")
    pickle_outputs = os.path.join(pickle_dir_path, "inputs.pkl")
    pickle_inputs_lengths = os.path.join(pickle_dir_path, "inputs_length.pkl")
    pickle_outputs_lengths = os.path.join(pickle_dir_path, "outputs_length.pkl")
    if io_b and iol_b:
        inputs = pickle.load(open(pickle_inputs, "rb"))
        outputs = pickle.load(open(pickle_outputs, "rb"))
        inputs_lengths = pickle.load(open(pickle_inputs_lengths, "rb"))
        outputs_lengths = pickle.load(open(pickle_outputs_lengths, "rb"))
        return inputs, outputs, inputs_lengths, outputs_lengths
    elif io_b:
        inputs = pickle.load(open(pickle_inputs, "rb"))
        outputs = pickle.load(open(pickle_outputs, "rb"))
        return inputs, outputs
    elif iol_b:
        inputs_lengths = pickle.load(open(pickle_inputs_lengths, "rb"))
        outputs_lengths = pickle.load(open(pickle_outputs_lengths, "rb"))
        return inputs_lengths, outputs_lengths
    else:
        log.info(
            "One of inputs-outputs and input-outputs-lengths must be true to load some pikcle"
        )


def load_word_freq(pickle_dir_path: str) -> dict:
    """It loads the word frequency dict

    Args:
        pickle_dir_path: pickle_dir_path: A path to the pickle dir

    Returns:
        word_freq: The word frequency dict

    """
    pickle_word_freq = os.path.join(pickle_dir_path, "word_freq.pkl")
    word_freq = pickle.load(open(pickle_word_freq, "rb"))

    return word_freq
