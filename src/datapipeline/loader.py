from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np

def generate_text_sequences(lines, pastWords, vocab):
    '''
    Generate lists of X_line and Y_line
    :param lines: list of str
    :param pastWords: ?
    :param vocab: ?
    :return: list of list of sequence (X_line), list of single token(Y_sequence)
    '''
    X_line = list()
    Y_line = list()
    pastWords = pastWords
    for line in lines:
        # Tokenize line
        lineTokenized = text_to_word_sequence(line, \
                                              filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r0123456789' + "'")
        # Get line length
        lengthLine = len(lineTokenized)
        lineBatch = lengthLine - pastWords

        # Substitute words outside vocab with <Unknown>
        for idx in range(0, len(lineTokenized)):
            if lineTokenized[idx] in vocab:
                continue
            else:
                lineTokenized[idx] = '<Unknown>'

        # Create sequences of text
        for i in range(0, lineBatch):
            # Window shifting
            X_sequence = lineTokenized[i:i + pastWords]
            X_line.append(X_sequence)
            # Word after shift
            Y_sequence = lineTokenized[i + pastWords]
            Y_line.append(Y_sequence)

    return (X_line, Y_line)


def batch_generator_data(batchsize, X_line, Y_line, embedding_dim, pastWords, embedded, vocab):
    embedding_dim = embedding_dim
    pastWords = pastWords
    x_batch = np.zeros(shape=(batchsize, pastWords, embedding_dim))
    y_batch = np.zeros(shape=(batchsize))

    while True:
        # Fill the batch with random continuous sequences of data.

        # Get a random start-index.
        # This points somewhere into the data.
        idx = np.random.randint(len(X_line) - batchsize)

        for i in range(0, batchsize):
            x_batch[i] = [embedded[vocab.index(x)] for x in X_line[idx + i]]
            y_batch[i] = vocab.index(Y_line[idx + i])

        # y_batch = to_categorical(y_batch, num_classes=len(vocab))

        yield (x_batch, y_batch)


def gen_vocab(data, max_tokens=200000):
    """
    helper function to generate the vocab for embedding.
    by default this will limit to the top 20000 tokens

    Parameters
    ----------
    data : dataset from the pipeline.

    Returns
    -------
    vocab :

    vectorizer : vectorizer for encoding x_train and y_train words


    """
    vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=200)
    text_ds = tf.data.Dataset.from_tensor_slices(data).batch(128)
    vectorizer.adapt(text_ds)
    vocab = vectorizer.get_vocabulary()
    return vocab, vectorizer