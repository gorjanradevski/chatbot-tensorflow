from typing import List
from utils.config import EOS_ID, PAD_ID
import tensorflow as tf


def pad_truncate_sequences(
    sequences: List[List], word2index: dict, max_length: int, input=True
) -> List[List]:
    """

    Args:
        sequences: List of sequences to be truncated or padded
        word2index: A dict that maps word = index
        max_length: Max length to perform padding or truncating

    Returns:
        polished_sequences: Prepared training/testing sequence

    """
    polished_sequences = []
    for sequence in sequences:
        polished_sequence = [word2index[word] for word in sequence]
        if input:
            if len(polished_sequence) >= max_length:
                polished_sequence[max_length - 1] = EOS_ID
            else:
                polished_sequence.append(EOS_ID)
        else:
            polished_sequence.insert(0, EOS_ID)

        polished_sequences.append(polished_sequence)

    polished_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        polished_sequences,
        maxlen=max_length,
        dtype="int32",
        padding="post",
        truncating="post",
        value=PAD_ID,
    )

    return polished_sequences
