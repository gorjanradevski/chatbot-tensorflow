from utils.pickles_utils import load_vocabs_pickles
from utils.data_processing_utils import prepare_input, replace_with_unk_chat
from utils.config import MAX_INPUT_OUTPUT_LENGTH, SEQ2SEQ_BATCH_SIZE
from utils.seq2seq_utils import pad_truncate_sequences

import tensorflow as tf
import sys
import os
import numpy as np
import argparse


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def load_model_parameters(
    meta_seq2seq_path: str, model_seq2seq_path: str, sess: tf.Session
):

    loader = tf.train.import_meta_graph(meta_seq2seq_path)
    loader.restore(sess, model_seq2seq_path)

    encoder_input = tf.get_default_graph().get_tensor_by_name(
        "Placeholders/encoder_inputs:0"
    )
    encoder_inputs_length = tf.get_default_graph().get_tensor_by_name(
        "Placeholders/encoder_inputs_length:0"
    )
    decoder_input = tf.get_default_graph().get_tensor_by_name(
        "Placeholders/decoder_inputs:0"
    )
    decoder_inputs_length = tf.get_default_graph().get_tensor_by_name(
        "Placeholders/decoder_inputs_length:0"
    )
    decoder_targets = tf.get_default_graph().get_tensor_by_name(
        "Placeholders/decoder_targets:0"
    )
    prediction = tf.get_default_graph().get_tensor_by_name("Decoder/prediction:0")
    is_training = tf.get_default_graph().get_tensor_by_name(
        "Placeholders/is_training:0"
    )

    return (
        encoder_input,
        encoder_inputs_length,
        decoder_input,
        decoder_inputs_length,
        decoder_targets,
        prediction,
        is_training,
    )


def predict_output(
    index2word,
    input_chat,
    encoder_input,
    encoder_inputs_length,
    decoder_input,
    decoder_inputs_length,
    decoder_targets,
    prediction,
    is_training,
    sess,
):

    enc_inputs = np.zeros((SEQ2SEQ_BATCH_SIZE, MAX_INPUT_OUTPUT_LENGTH))
    enc_inputs[0, :] = input_chat
    enc_len = np.zeros(SEQ2SEQ_BATCH_SIZE)
    enc_len[0] = len(input_chat)

    feed_dict = {
        encoder_input: enc_inputs,
        encoder_inputs_length: enc_len,
        decoder_input: [[]],
        decoder_inputs_length: [],
        decoder_targets: [[]],
        is_training: False,
    }

    indexes_of_words = sess.run(prediction, feed_dict=feed_dict)
    sentence = [index2word[index] for index in indexes_of_words[0]]

    return sentence


def start_chatting(args):
    """Performs the chatting with the user. It assumes that the pickles are in pickles

    Returns:

    """
    word2index, index2word = load_vocabs_pickles(
        args.pickle_dir, word2index_b=True, index2word_b=True
    )

    meta_seq2seq_path = os.path.join(args.log_dir, "seq2seq.meta")
    model_seq2seq_path = os.path.join(args.log_dir, "seq2seq")

    with tf.Session() as sess:

        encoder_input, encoder_inputs_length, decoder_input, decoder_inputs_length, decoder_targets, prediction, is_training = load_model_parameters(
            meta_seq2seq_path, model_seq2seq_path, sess
        )

        print("Chat started! For exit type: Bye!")

        while True:

            print("User says:")

            line = sys.stdin.readline()

            if line.strip().lower() == "bye!":
                print("Bot says: Bye!")
                break

            input_chat_raw = prepare_input(line)
            input_chat_clean = replace_with_unk_chat(input_chat_raw, word2index)
            polished_input = pad_truncate_sequences(
                [input_chat_clean], word2index, MAX_INPUT_OUTPUT_LENGTH, input=True
            )
            print(polished_input[0])

            reply = predict_output(
                index2word,
                polished_input,
                encoder_input,
                encoder_inputs_length,
                decoder_input,
                decoder_inputs_length,
                decoder_targets,
                prediction,
                is_training,
                sess,
            )

            print("Bot says:\n", " ".join(reply))

            print("----------------------------------------------")


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

    parser.add_argument(
        "--log_dir", type=str, default="../logs/", help="Path where the log dir is"
    )

    start_chatting(parser.parse_args())
