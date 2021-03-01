import argparse
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# from IPython import get_ipython

# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.modelling.model import gen_vocab, generate_text_sequences
from src.datapipeline.datapipeline import Datapipeline

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# %%
def initialize(data_path, model_path):
    dpl = Datapipeline(data_path)
    dpl.transform()
    train, val = dpl.split_data()
    model = load_model(model_path)
    vocab, vectorizer = gen_vocab(train.values)
    
    return train, val, model, vocab, vectorizer


# %%
def tokenize(val, vocab, vectorizer):
    
    x_val, y_val = generate_text_sequences(val.values, 5, vocab)
    x_input_val = vectorizer.apply(list(map(lambda x: ' '.join(x), x_val)))
    x_input_val = tf.gather(x_input_val, [0,1,2,3,4], axis=1)
    
    return x_input_val


# %%
def generate_next_word(phrase, vectorizer, model, vocab):
    # if phrase is list of tokens
    if type(phrase) == list:
        x_input_val = vectorizer.apply([' '.join(phrase)])
    # if phrase is string
    elif type(phrase) == str:
        x_input_val = vectorizer.apply([phrase])
    x_input_val = tf.gather(x_input_val, [0,1,2,3,4], axis=1)
    prob_ = model.predict(x_input_val)
    idx = np.argmax(prob_)
    return vocab[idx]

# generate_single_tweet('the trade deficit rose to')


# %%
def generate_tweet(phrase, max_char=140):
    word = phrase[-1]
    char_count = 0
    tweet_range = np.random.randint(15,25)
    while word != '' and char_count <= max_char and len(phrase) < tweet_range:
        input_phrase = phrase[-5:]
        word = generate_next_word(input_phrase, vectorizer, model, vocab)
        if word == '':
            break
        elif word =='a' and input_phrase[-1] == 's' and input_phrase[-2] == 'u':
            word = 'usa'
            phrase.pop()
            phrase.pop()

        phrase.append(word)
        char_count = len(' '.join(phrase))
        
    return ' '.join(phrase)


# %%

def run_gen(sentence):
    # x_input_val = tokenize(val, vocab, vectorizer)
    global train
    global val
    global model
    global vocab
    global vectorizer
    train, val, model, vocab, vectorizer = initialize('./data/raw/realdonaldtrump.csv', './model/trump_bot_bidirstack.h5')
    phrase = sentence.split()[:5] # ['the', 'white', 'house', 'our', 'country']
    print(generate_tweet(phrase))
    return generate_tweet(phrase)

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sen', '--sentence', type=str,
                        default="white house is our first",
                        # default="/Users/jufri/Gdrive_AIAP/all-assignments/assignment7/data/nasi_ayam_1.jpg",
                        help='Path to image')

    args = parser.parse_args()
    sentence = args.sentence
    run_gen(sentence)