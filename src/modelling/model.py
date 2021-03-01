import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.preprocessing.text import text_to_word_sequence

def loadGloveModel(gloveFile):
    """
    loads GloVe model

    Parameters
    ----------
    gloveFile : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    model = {}
    with open(gloveFile, encoding="utf8") as f:
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
    return model

def gen_vocab(data, max_tokens = 200000):
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


def embed_matrix(embedding_model, vocab, embedding_dim = 100):
    """
    embedding for the given vocab fed in 

    Parameters
    ----------
    embedding_model : TYPE
        DESCRIPTION.
    vocab : TYPE
        DESCRIPTION.
    embedding_dim : TYPE, optional
        DESCRIPTION. The default is 100.

    Returns
    -------
    None.

    """
    num_tokens = len(vocab) + 2
    hits = 0
    misses = 0
    word_index = dict(zip(vocab, range(2, len(vocab))))
    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embedding_model.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))
    
    embedding_layer = Embedding(num_tokens,
                                embedding_dim,
                                embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                                trainable=False)
    return embedding_layer

def generate_text_sequences(lines, pastWords, vocab):
    X_line = list()
    Y_line = list()
    pastWords = pastWords
    for line in lines:
        # Tokenize line
        lineTokenized = text_to_word_sequence(line.item(),\
                                              filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r0123456789'+"'")
        #Get line length
        lengthLine = len(lineTokenized)
        lineBatch = lengthLine - pastWords
        
        # Substitute words outside vocab with <Unknown>
        for idx in range(0,len(lineTokenized)):
            if lineTokenized[idx] in vocab:
                continue
            else:
                lineTokenized[idx] = '<Unknown>'
        
        #Create sequences of text
        for i in range(0,lineBatch):
            X_sequence = lineTokenized[i:i+pastWords]
            X_line.append(X_sequence)
            Y_sequence = lineTokenized[i+pastWords]
            Y_line.append(Y_sequence)
    
    return(X_line, Y_line)



class twitter_model():
    
    def __init__(self, glove_path):
        # do locally save the glove.txt into the directory as given
        # will be changed once this is moved to polyaxon notebook
        
        self.glovemodel = loadGloveModel(glove_path)
        self.vocab = None
        self.vectorizer = None
        self.model = None

    def build_model(self, data):
        """
        data: ?
        Parameters
        ----------
        vocab : 
            the vocab of the vectorise
        
        Returns
        ---------
        Prints a summary of the model created.
        self.model instantiated to be used on train method

        """
        # Build vocab
        self.vocab, self.vectorizer = gen_vocab(data)

        # build embedding layer from glove 
        int_sequences_input = keras.Input(shape=(None,), dtype="int64")

        embedding_layer = embed_matrix(embedding_model=self.glovemodel, vocab=self.vocab, embedding_dim = 100)
        embedded_sequences = embedding_layer(int_sequences_input)
        # For stacked
        x = layers.Bidirectional(layers.LSTM(100, return_sequences=True))(embedded_sequences)
        x = layers.LSTM(100)(x)
        # For single layer bidir
        # x = layers.Bidirectional(layers.LSTM(100))(embedded_sequences)
        # For basic model
        # x = layers.LSTM(100)(embedded_sequences)
        preds = layers.Dense(len(self.vocab), activation = 'softmax')(x)
        self.model = keras.Model(inputs=int_sequences_input, outputs=preds)
        self.model.compile(loss="sparse_categorical_crossentropy", 
                           optimizer="rmsprop", 
                           metrics=["acc"])
        self.model.summary()
    
    def get_train_data(self, data):
        # call in the generate_text_sequences function to get the x_train and y_train 
        # alternative can call the generate_text_sequences function in train method directly to skip this function
        self.x_train, self.y_train = generate_text_sequences(lines=data, pastWords=5, vocab=self.vocab)
        self.x_train = self.vectorizer.apply(list(map(lambda x: ' '.join(x), self.x_train)))
        self.x_train = tf.gather(self.x_train, [0,1,2,3,4], axis=1)
        self.y_train = self.vectorizer.apply(self.y_train)
        self.y_train = tf.gather(self.y_train, [0], axis=1)
    
    def train(self):
        """
        trains the model
        requires to build the model first

        Returns
        -------
        None.

        """
        # self.vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=200)

        # below - generate the text sequence to be fed into the model
        # x_train = self.vectorizer(np.array([[s] for s in train_samples])).numpy()
        # y_train = self.vectorizer(np.array([[s] for s in val_samples])).numpy()
        
        self.model.fit(self.x_train, self.y_train, batch_size=128, epochs=5)
        return self.model # for saving
    
    def get_data(self):
        return self.x_train, self.y_train
    
    def get_vocab(self):
        return self.vocab, self.vectorizer
