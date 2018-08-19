from utils.config import (
    SKIPGRAM_BATCH_SIZE,
    NUM_SAMPLED,
    EMBEDDING_SIZE,
    SKIPGRAM_EPOCHS,
    VOCAB_SIZE,
    SKIPGRAM_LR,
    WINDOW_SIZE,
)

import pickle
import numpy as np
import tensorflow as tf
import math
import os
from typing import Tuple
from tqdm import tqdm
import logging
import argparse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def add_pair_to_train(words: list, skipgram_train_data: list, word2index: dict) -> list:
    """Given a list of words it adds pairs of train data to the dataset

    Args:
        words: A list of words
        skipgram_train_data: Train data so far
        word2index: A dict of word = index pairs

    Returns:
        skipgram_train_data_new: Updated train data

    """
    skipgram_train_data_new = skipgram_train_data
    for word_index, word in enumerate(words):
        for nb_word in words[
            max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(words))
            + 1
        ]:
            if nb_word != word:
                skipgram_train_data_new.append([word2index[word], word2index[nb_word]])

    return skipgram_train_data_new


def create_skipgram_train_data(args) -> np.array:
    """Prepares the training data for the skipgram model

    Args:
        args: Command line args or just the pickle dir argument

    Returns:
        skipgram_train_data: Pairs of words

    """
    # Prepare paths
    input_sentences_path = os.path.join(args.pickle_dir, "inputs.pkl")
    output_sentences_path = os.path.join(args.pickle_dir, "outputs.pkl")
    word2index_path = os.path.join(args.pickle_dir, "word2index.pkl")

    # Load pickles
    input_sentences = pickle.load(open(input_sentences_path, "rb"))
    output_sentences = pickle.load(open(output_sentences_path, "rb"))
    word2index = pickle.load(open(word2index_path, "rb"))

    skipgram_train_data = []
    log.info("Preparing the skipgram training data")
    for input_sen, output_sen in tqdm(zip(input_sentences, output_sentences)):

        skipgram_train_data = add_pair_to_train(
            input_sen, skipgram_train_data, word2index
        )
        skipgram_train_data = add_pair_to_train(
            output_sen, skipgram_train_data, word2index
        )

    return np.array(skipgram_train_data)


def generate_batch(skipgram_train_data: np.array, step: int) -> Tuple:
    """

    Args:
        skipgram_train_data: The skipgram dataset
        step: Current step

    Returns:
        inputs: Input word
        label: Label word

    """
    inputs = np.ndarray(shape=(SKIPGRAM_BATCH_SIZE), dtype=np.int32)
    labels = np.ndarray(shape=(SKIPGRAM_BATCH_SIZE, 1), dtype=np.int32)

    inputs[:] = skipgram_train_data[
        step * SKIPGRAM_BATCH_SIZE : (step + 1) * SKIPGRAM_BATCH_SIZE, 0
    ]
    labels[:, 0] = skipgram_train_data[
        step * SKIPGRAM_BATCH_SIZE : (step + 1) * SKIPGRAM_BATCH_SIZE, 1
    ]

    return inputs, labels


def build_skipgram():
    """Builds the Skipgram model

    Returns:
        inputs_placeholder: Placeholder used to pour the inputs
        labels_placeholder: Placeholder used to pour the labels
        loss_fun: The loss used to compute the model loss
        optimizer: The optimizer used to update the weights
        embeddings: The weight matrix that will be saved

    """
    tf.reset_default_graph()

    inputs_placeholder = tf.placeholder(tf.int32, shape=[None], name="Inputs")
    labels_placeholder = tf.placeholder(tf.int32, shape=[None, 1], name="Labels")

    with tf.variable_scope("word2vec"):

        embeddings = tf.get_variable(
            name="embeddings", shape=(VOCAB_SIZE, EMBEDDING_SIZE), trainable=True
        )

        batch_embeddings = tf.nn.embedding_lookup(embeddings, inputs_placeholder)

        weights = tf.Variable(
            tf.truncated_normal(
                [VOCAB_SIZE, EMBEDDING_SIZE], stddev=1.0 / math.sqrt(EMBEDDING_SIZE)
            )
        )

        biases = tf.Variable(tf.zeros([VOCAB_SIZE]))

        loss_fun = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=weights,
                biases=biases,
                labels=labels_placeholder,
                inputs=batch_embeddings,
                num_sampled=NUM_SAMPLED,
                num_classes=VOCAB_SIZE,
            )
        )

        optimizer = tf.train.GradientDescentOptimizer(SKIPGRAM_LR).minimize(loss_fun)

    log.info("Skipgram model built")

    return inputs_placeholder, labels_placeholder, loss_fun, optimizer, embeddings


def run(
    inputs_placeholder,
    labels_placeholder,
    loss_fun,
    optimizer,
    embeddings,
    skipgram_train_data,
):
    """Learns and saves the embeddings matrix

    Args:
        inputs_placeholder: Placeholder used to pour the inputs
        labels_placeholder: Placeholder used to pour the labels
        loss_fun: The loss used to compute the model loss
        optimizer: The optimizer used to update the weights
        embeddings: The weight matrix that will be saved
        skipgram_train_data: The skipgram training data

    Returns:

    """

    init = tf.global_variables_initializer()
    saver = tf.train.Saver({"embeddings": embeddings})

    writer = tf.summary.FileWriter("../logs", tf.get_default_graph())
    with tf.Session() as sess:

        sess.run(init)

        num_steps = len(skipgram_train_data) // SKIPGRAM_BATCH_SIZE

        for e in range(SKIPGRAM_EPOCHS):
            # Shuffle training data
            np.random.shuffle(skipgram_train_data)
            epoch_loss = 0
            for step in tqdm(range(num_steps)):
                inputs, labels = generate_batch(skipgram_train_data, step)

                feed_dict = {inputs_placeholder: inputs, labels_placeholder: labels}
                _, loss_val = sess.run([optimizer, loss_fun], feed_dict)

                epoch_loss += loss_val

            log.info(f"Loss after epoch {e+1} is: {epoch_loss}")

        save_path = saver.save(sess, "../logs/skipgram")
        log.info(f"Model saved in path: {save_path}")

    writer.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="The script takes the pickle directory location and trains the Skipgram model"
        "and saves the embedding matrix"
    )

    parser.add_argument(
        "--pickle_dir",
        type=str,
        default="../pickles/",
        help="Path where the pickle dir is",
    )

    skipgram_train_data = create_skipgram_train_data(parser.parse_args())
    inputs_placeholder, labels_placeholder, loss_fun, optimizer, embeddings = (
        build_skipgram()
    )

    run(
        inputs_placeholder,
        labels_placeholder,
        loss_fun,
        optimizer,
        embeddings,
        skipgram_train_data,
    )
