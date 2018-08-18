from utils.data_processing_utils import (
    prepare_input,
    prepare_movie_line,
    replace_with_unk,
    prepare_movie_conv,
)

import tqdm
import pickle
import argparse
import logging
import os

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
        for movie_line_raw in tqdm.tqdm(movie_lines):
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
        A path where the movie_lines.txt file is
        word_freq: A dict as word = freq

    Returns:
        uterances: A dict as id = movie line

    """
    dialogs = {}
    with open(movie_lines_path, "rb") as movie_lines:
        for movie_line_row in tqdm.tqdm(movie_lines):
            id, movie_line_clean = prepare_movie_line(movie_line_row)
            raw_words = prepare_input(movie_line_clean)
            unked_words = replace_with_unk(raw_words, word_freq)
            dialogs[id] = " ".join(unked_words)

    return dialogs


def design_inputs_outputs(args) -> None:
    """Creates the input and output pickles

    Args:
        args: Arguments passed through the command line

    Returns:

    """
    mov_lines_path = os.path.join(args.data_dir, "movie_lines.txt")
    mov_conv_path = os.path.join(args.data_dir, "movie_conversations.txt")
    pickle_inputs = os.path.join(args.pickle_dir, "inputs.pkl")
    pickle_outputs = os.path.join(args.pickle_dir, "outputs.pkl")

    inputs = []
    outputs = []
    word_freq = compute_word_frequencies(mov_lines_path)
    dialogs = prepare_uterances(mov_lines_path, word_freq)

    with open(mov_conv_path, "rb") as mov_conv_file:
        for movie_conv_row in tqdm.tqdm(mov_conv_file):
            conversations = prepare_movie_conv(movie_conv_row)
            for c in range(len(conversations) - 1):
                inputs.append(dialogs[conversations[c]])
                outputs.append(dialogs[conversations[c + 1]])

    pickle.dump(inputs, open(pickle_inputs, "wb"))
    pickle.dump(outputs, open(pickle_outputs, "wb"))


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
        "--pickle_dir", type=str, default="data/", help="Path where the data is"
    )

    design_inputs_outputs(parser.parse_args())
