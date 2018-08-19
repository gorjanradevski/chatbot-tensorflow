from utils.config import (
    SEQ2SEQ_EPOCHS,
    SEQ2SEQ_LR,
    ENCODER_DECODER_HIDDEN_UNITS,
    SEQ2SEQ_BATCH_SIZE,
    NUM_LAYERS,
    VOCAB_SIZE,
    MAX_INPUT_OUTPUT_LENGTH,
    EMBEDDING_SIZE,
)

from utils.pickles_utils import load_vocabs_pickles, load_input_output_lengths_pickles
from utils.seq2seq_utils import pad_truncate_sequences

import tensorflow as tf
import os
from typing import Tuple
from tqdm import trange
import argparse
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def create_training_data(args) -> Tuple:
    """

    Args:
        args: A path to the pickle directory

    Returns:
        training_data_encoder: The encoder inputs
        training_data_encoder_l: The encoder inputs lengths
        training_data_decoder: The decoder inputs
        training_data_decoder_l: The decoder inputs lengths
        training_data_decoder_targets: The decoder targets

    """
    inputs, outputs, training_data_encoder_l, training_data_decoder_l = load_input_output_lengths_pickles(
        args.pickle_dir, io_b=True, iol_b=True
    )

    word2index = load_vocabs_pickles(
        args.pickle_dir, word2index_b=True, index2word_b=False
    )

    training_data_encoder = pad_truncate_sequences(
        inputs, word2index, MAX_INPUT_OUTPUT_LENGTH, input=True
    )
    training_data_decoder = pad_truncate_sequences(
        outputs, word2index, MAX_INPUT_OUTPUT_LENGTH, input=True
    )
    training_data_decoder_targets = pad_truncate_sequences(
        outputs, word2index, MAX_INPUT_OUTPUT_LENGTH, input=False
    )

    return (
        training_data_encoder,
        training_data_encoder_l,
        training_data_decoder,
        training_data_decoder_l,
        training_data_decoder_targets,
    )


def generate_batch(
    training_data_encoder,
    training_data_encoder_l,
    training_data_decoder,
    training_data_decoder_l,
    training_data_decoder_targets,
    step,
):
    train_encoder = training_data_encoder[
        step * SEQ2SEQ_BATCH_SIZE : (step + 1) * SEQ2SEQ_BATCH_SIZE
    ]
    train_encoder_l = training_data_encoder_l[
        step * SEQ2SEQ_BATCH_SIZE : (step + 1) * SEQ2SEQ_BATCH_SIZE
    ]
    train_decoder = training_data_decoder[
        step * SEQ2SEQ_BATCH_SIZE : (step + 1) * SEQ2SEQ_BATCH_SIZE
    ]
    train_decoder_l = training_data_decoder_l[
        step * SEQ2SEQ_BATCH_SIZE : (step + 1) * SEQ2SEQ_BATCH_SIZE
    ]
    train_decoder_t = training_data_decoder_targets[
        step * SEQ2SEQ_BATCH_SIZE : (step + 1) * SEQ2SEQ_BATCH_SIZE
    ]

    return (
        train_encoder,
        train_encoder_l,
        train_decoder,
        train_decoder_l,
        train_decoder_t,
    )


def lstm_cell():
    return tf.nn.rnn_cell.LSTMCell(ENCODER_DECODER_HIDDEN_UNITS)


def build_graph() -> Tuple:
    """Builds the sequence2sequence models

    Returns:
        encoder_inputs: Encoder inputs placeholder
        encoder_inputs_length: Encoder inputs length placeholder
        decoder_inputs: Decoder inputs placeholder
        decoder_inputs_length: Decoder inputs lengths placeholder
        decoder_targets: Decoder targets placeholder
        is_training: Training or no training placeholder
        loss_fun: Loss function
        train_opt: Training optimizer

    """
    tf.reset_default_graph()

    with tf.variable_scope("Placeholders"):

        encoder_inputs = tf.placeholder(
            shape=(None, None), dtype=tf.int32, name="encoder_inputs"
        )
        encoder_inputs_length = tf.placeholder(
            shape=(None,), dtype=tf.int32, name="encoder_inputs_length"
        )
        decoder_inputs = tf.placeholder(
            shape=(None, None), dtype=tf.int32, name="decoder_inputs"
        )
        decoder_inputs_length = tf.placeholder(
            shape=(None,), dtype=tf.int32, name="decoder_inputs_length"
        )
        decoder_targets = tf.placeholder(
            shape=(None, None), dtype=tf.int32, name="decoder_targets"
        )
        is_training = tf.placeholder(dtype=tf.bool, name="is_training")

    with tf.variable_scope("Embeddings"):

        embeddings = tf.Variable(
            tf.random_uniform([VOCAB_SIZE, EMBEDDING_SIZE], -1.0, 1.0),
            dtype=tf.float32,
            trainable=True,
            name="embeddings",
        )

        # TODO: Train skipgram to obtain embeddings
        # embedding_saver = tf.train.Saver({"embeddings": embeddings})
        # with tf.Session() as sess:
        # embedding_saver.restore(sess, "logs/skipgram")

        encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
        decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

    with tf.variable_scope("Encoder"):

        stacked_lstm_encoder = tf.contrib.rnn.MultiRNNCell(
            [lstm_cell() for _ in range(NUM_LAYERS)]
        )

        # noinspection PyUnresolvedReferences
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            stacked_lstm_encoder,
            encoder_inputs_embedded,
            sequence_length=encoder_inputs_length,
            dtype=tf.float32,
            time_major=False,
        )

    with tf.variable_scope("Decoder"):

        stacked_lstm_decoder = tf.contrib.rnn.MultiRNNCell(
            [lstm_cell() for _ in range(NUM_LAYERS)]
        )

        W = tf.Variable(
            tf.random_uniform([ENCODER_DECODER_HIDDEN_UNITS, VOCAB_SIZE], -1, 1),
            dtype=tf.float32,
        )
        b = tf.Variable(tf.zeros([VOCAB_SIZE]), dtype=tf.float32)

        def loop_fn(time, cell_output, cell_state, loop_state):

            ended = tf.constant(1, shape=(SEQ2SEQ_BATCH_SIZE,), dtype=tf.int64)

            def testing_end():
                def expired_time():
                    return tf.constant(True, shape=(SEQ2SEQ_BATCH_SIZE,), dtype=tf.bool)

                def end_output():
                    output_logits = tf.add(tf.matmul(cell_output, W), b)
                    prediction = tf.argmax(output_logits, axis=1)
                    return tf.equal(prediction, ended)

                if cell_output is None:
                    return tf.constant(
                        False, shape=(SEQ2SEQ_BATCH_SIZE,), dtype=tf.bool
                    )

                alter_finish = tf.greater_equal(time, MAX_INPUT_OUTPUT_LENGTH)

                return tf.cond(
                    alter_finish, lambda: expired_time(), lambda: end_output()
                )

            def training_end():
                elements_finished = time >= decoder_inputs_length
                return elements_finished

            def get_next_input_train(time):
                embedded_value = decoder_inputs_embedded[:, time, :]
                embedded_value.set_shape([SEQ2SEQ_BATCH_SIZE, EMBEDDING_SIZE])

                return embedded_value

            def get_next_input_prediction():
                output_logits = tf.add(tf.matmul(cell_output, W), b)
                prediction = tf.argmax(output_logits, axis=1)
                embedded_value = tf.nn.embedding_lookup(embeddings, prediction)

                return embedded_value

            def get_next_input(time):

                return tf.cond(
                    is_training,
                    lambda: get_next_input_train(time),
                    lambda: get_next_input_prediction(),
                )

            emit_output = cell_output  # == None for time == 0
            next_loop_state = None

            elements_finished = tf.cond(
                is_training, lambda: training_end(), lambda: testing_end()
            )

            finished = tf.reduce_all(elements_finished)

            if cell_output is None:  # time == 0
                next_cell_state = encoder_state
                next_input = tf.nn.embedding_lookup(
                    embeddings, tf.ones([SEQ2SEQ_BATCH_SIZE], dtype=tf.int32)
                )
            else:
                next_cell_state = cell_state
                next_input = tf.cond(
                    finished,
                    lambda: tf.nn.embedding_lookup(
                        embeddings, tf.zeros([SEQ2SEQ_BATCH_SIZE], dtype=tf.int32)
                    ),
                    lambda: get_next_input(time),
                )

            return (
                elements_finished,
                next_input,
                next_cell_state,
                emit_output,
                next_loop_state,
            )

        decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(
            stacked_lstm_decoder, loop_fn
        )

        decoder_outputs = decoder_outputs_ta.stack()

        decoder_outputs_flat = tf.reshape(
            decoder_outputs, (-1, ENCODER_DECODER_HIDDEN_UNITS)
        )

        decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)

        decoder_logits = tf.reshape(
            decoder_logits_flat, (-1, SEQ2SEQ_BATCH_SIZE, VOCAB_SIZE)
        )

        batch_time = tf.shape(decoder_logits)[0]

        batch_logits = tf.transpose(decoder_logits, [1, 0, 2])

        prediction = tf.argmax(batch_logits, 2, name="prediction")

    with tf.variable_scope("loss_optimization"):

        target_weights = tf.sequence_mask(
            decoder_inputs_length, maxlen=batch_time, dtype=decoder_logits.dtype
        )

        batch_targets = tf.slice(
            decoder_targets, [0, 0], [SEQ2SEQ_BATCH_SIZE, batch_time]
        )

        stepwise_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=batch_targets, logits=batch_logits
        )

        loss_fun = tf.reduce_sum(
            tf.multiply(stepwise_cross_entropy, target_weights) / SEQ2SEQ_BATCH_SIZE
        )

        optimizer = tf.train.AdamOptimizer(SEQ2SEQ_LR).minimize(loss_fun)

    return (
        encoder_inputs,
        encoder_inputs_length,
        decoder_inputs,
        decoder_inputs_length,
        decoder_targets,
        is_training,
        loss_fun,
        optimizer,
        prediction,
    )


def train_model(
    encoder_placeholder,
    encoder_length_placeholder,
    decoder_placeholder,
    decoder_length_placeholder,
    decoder_targets_placeholder,
    is_training_placeholder,
    loss_fun,
    optimizer,
    prediction,
    input_data,
    input_data_length,
    output_data,
    output_data_length,
    output_target_data,
):
    """Given all placeholders, loss_fun, optimizer and inputs it trains the model

    Args:
        encoder_placeholder:
        encoder_length_placeholder:
        decoder_placeholder:
        decoder_length_placeholder:
        decoder_targets_placeholder:
        is_training_placeholder:
        loss_fun:
        optimizer:
        input_data:
        input_data_length:
        output_data:
        output_data_length:
        output_target_data:

    Returns:

    """
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()

        num_steps = len(input_data) // SEQ2SEQ_BATCH_SIZE

        print("The total number of steps are: ", num_steps)

        for e in range(SEQ2SEQ_EPOCHS):

            epoch_loss = 0
            train_for_epoch = trange(num_steps)

            for step in train_for_epoch:
                input_batch, input_batch_length, output_batch, output_batch_length, output_batch_targets = generate_batch(
                    input_data,
                    input_data_length,
                    output_data,
                    output_data_length,
                    output_target_data,
                    step,
                )

                feed_dict = {
                    encoder_placeholder: input_batch,
                    encoder_length_placeholder: input_batch_length,
                    decoder_placeholder: output_batch,
                    decoder_length_placeholder: output_batch_length,
                    decoder_targets_placeholder: output_batch_targets,
                    is_training_placeholder: True,
                }

                _, loss = sess.run([optimizer, loss_fun], feed_dict)
                epoch_loss += loss

                train_for_epoch.set_description(
                    f"Current step loss: %g" % loss
                )

        save_path = saver.save(sess, "logs/seq2seq")
        print(f"Model saved in path {save_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="The script takes the pickle directory location and trains the Seq model"
        "and saves the model"
    )

    parser.add_argument(
        "--pickle_dir",
        type=str,
        default="../pickles/",
        help="Path where the pickle dir is",
    )

    training_data_encoder, training_data_encoder_l, training_data_decoder, training_data_decoder_l, training_data_decoder_targets = create_training_data(
        parser.parse_args()
    )

    encoder_placeholder, encoder_length_placeholder, decoder_placeholder, decoder_length_placeholder, decoder_targets_placeholder, is_training_placeholder, loss_fun, optimizer, prediction = (
        build_graph()
    )

    print("Graph built...")

    train_model(
        encoder_placeholder,
        encoder_length_placeholder,
        decoder_placeholder,
        decoder_length_placeholder,
        decoder_targets_placeholder,
        is_training_placeholder,
        loss_fun,
        optimizer,
        prediction,
        training_data_encoder,
        training_data_encoder_l,
        training_data_decoder,
        training_data_decoder_l,
        training_data_decoder_targets,
    )
