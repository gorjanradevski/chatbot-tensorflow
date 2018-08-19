from utils.pickles_utils import load_vocabs_pickles, load_word_freq
from utils.data_processing_utils import prepare_input, replace_with_unk
from utils.config import MAX_INPUT_OUTPUT_LENGTH, SEQ2SEQ_BATCH_SIZE
from utils.seq2seq_utils import pad_truncate_sequences

import tensorflow as tf
import sys
import os
import numpy as np
import argparse


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def prepare_output(
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

    sentence = []

    for index in indexes_of_words[0]:

        sentence.append(index2word[index])

    return sentence


def start_chatting(args):
    """Performs the chatting with the user. It assumes that the pickles are in pickles

    Returns:

    """
    word2index, index2word = load_vocabs_pickles(
        args.pickle_dir, word2index_b=True, index2word_b=True
    )
    word_freq = load_word_freq(args.pickle_dir)

    meta_seq2seq_path = os.path.join(args.log_dir, "seq2seq.meta")
    model_seq2seq_path = os.path.join(args.log_dir, "seq2seq")

    with tf.Session() as sess:

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

        print(
            "Chat started, please keep the input below 20 words, for exit type --- Bye! ---"
        )

        while True:

            print("User says:")

            line = sys.stdin.readline()

            if line.strip().lower() == "bye!":
                print("Bot says: Bye!")
                break

            input_chat_raw = prepare_input(line)
            input_chat_clean = replace_with_unk(input_chat_raw, word_freq)
            polished_input = pad_truncate_sequences(
                input_chat_clean, word2index, MAX_INPUT_OUTPUT_LENGTH, input=True
            )

            output = prepare_output(
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

            print("Bot says:\n", " ".join(output))

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
        "--log_dir",
        type=str,
        default="../logs/",
        help="Path where the log dir is",
    )

    start_chatting(parser.parse_args())
