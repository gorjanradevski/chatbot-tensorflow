from utils.data_processing_utils import (
    prepare_input,
    prepare_movie_line,
    replace_with_unk,
    prepare_movie_conv,
)

from utils.config import PAD_ID, EOS_ID, UNK_ID

from tqdm import tqdm
import pickle
import argparse
import logging
import os
from typing import Tuple

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


def compute_word_frequencies(movie_lines_path: str) -> dict:
    """Computes the word frequencies of all words

    Args:
        movie_lines_path: A path where the movie_lines.txt file is

    Returns:
        word_freq: A dictonary that contains the word frequencies

    """
    word_freq = {}
    with open(movie_lines_path, "rb") as movie_lines:
        for movie_line_raw in tqdm(movie_lines):
            _, movie_line_clean = prepare_movie_line(movie_line_raw)
            raw_words = prepare_input(movie_line_clean)
            for word in raw_words:
                if word not in word_freq:
                    word_freq[word] = 0
                else:
                    word_freq[word] += 1

    return word_freq


def prepare_uterances(movie_lines_path: str, word_freq: dict) -> dict:
    """Prepares a dict as id = movie line

    Args:
        movie_lines_path: A path where the movie_lines.txt file is
        word_freq: A dict as word = freq

    Returns:
        dialogs: A dict as id = movie line
        dialogs_lengths: A dict as id = length of movie line

    """
    dialogs = {}
    lengths = {}
    with open(movie_lines_path, "rb") as movie_lines:
        for movie_line_row in tqdm(movie_lines):
            id, movie_line_clean = prepare_movie_line(movie_line_row)
            raw_words = prepare_input(movie_line_clean)
            unked_words = replace_with_unk(raw_words, word_freq)
            dialogs[id] = unked_words

    return dialogs


def build_vocab(movie_lines_path: str, word_freq: dict) -> Tuple:
    """Builds the vocabulary

    Args:
        movie_lines_path: A path where the movie_lines.txt file is
        word_freq: A dict as word = freq

    Returns:
        word2index: A dict that maps word = index
        index2word: A dict that maps index = word

    """
    word2index = {"<pad>": PAD_ID, "<end>": EOS_ID, "<unk>": UNK_ID}
    index = 0
    with open(movie_lines_path, "rb") as movie_lines:
        for movie_line_row in tqdm(movie_lines):
            _, movie_line_clean = prepare_movie_line(movie_line_row)
            raw_words = prepare_input(movie_line_clean)
            unked_words = replace_with_unk(raw_words, word_freq)
            for word in unked_words:
                if word not in word2index:
                    word2index[word] = index
                    index += 1

    index2word = dict(zip(word2index.values(), word2index.keys()))

    return word2index, index2word


def design_inputs_outputs_vocab(args) -> None:
    """Creates the input and output pickles

    Args:
        args: Arguments passed through the command line

    Returns:

    """
    mov_lines_path = os.path.join(args.data_dir, "movie_lines.txt")
    mov_conv_path = os.path.join(args.data_dir, "movie_conversations.txt")
    pickle_inputs = os.path.join(args.pickle_dir, "inputs.pkl")
    pickle_outputs = os.path.join(args.pickle_dir, "outputs.pkl")
    pickle_inputs_length = os.path.join(args.pickle_dir, "inputs_length.pkl")
    pickle_outputs_length = os.path.join(args.pickle_dir, "outputs_length.pkl")
    pickle_word2index = os.path.join(args.pickle_dir, "word2index.pkl")
    pickle_index2word = os.path.join(args.pickle_dir, "index2word.pkl")

    # Compute input and outputs pickles
    inputs = []
    outputs = []
    # Compute inputs length and outputs length
    inputs_lengths = []
    outputs_lengths = []
    word_freq = compute_word_frequencies(mov_lines_path)
    dialogs = prepare_uterances(mov_lines_path, word_freq)

    with open(mov_conv_path, "rb") as mov_conv_file:
        for movie_conv_row in tqdm(mov_conv_file):
            conversations = prepare_movie_conv(movie_conv_row)
            for c in range(len(conversations) - 1):
                input = dialogs[conversations[c]]
                output = dialogs[conversations[c + 1]]
                input_length = len(input)
                output_length = len(output)
                inputs.append(input)
                outputs.append(output)
                inputs_lengths.append(input_length)
                outputs_lengths.append(output_length)


    pickle.dump(inputs, open(pickle_inputs, "wb"))
    pickle.dump(outputs, open(pickle_outputs, "wb"))
    pickle.dump(inputs_lengths, open(pickle_inputs_length, "wb"))
    pickle.dump(outputs_lengths, open(pickle_outputs_length, "wb"))
    log.info(f"Input and output pickles dumped at location {args.pickle_dir}")

    # Compute word2index and index2word pickles
    word2index, index2word = build_vocab(mov_lines_path, word_freq)
    pickle.dump(word2index, open(pickle_word2index, "wb"))
    pickle.dump(index2word, open(pickle_index2word, "wb"))
    log.info(f"Word2index and index2word pickles dumped at location {args.pickle_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="The scripts takes the data path and a pickle directory path, "
        "performs all necessary preparation and dumps the input and output"
        "pickles."
    )

    parser.add_argument(
        "--data_dir", type=str, default="data/", help="Path where the data is"
    )

    parser.add_argument(
        "--pickle_dir", type=str, default="pickles/", help="Path where the pickle dir is"
    )

    design_inputs_outputs_vocab(parser.parse_args())
