# Boilerplate code to serve a web app
from flask import Flask, request, render_template, session, redirect, flash
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import load_model

from src.inference import *

app = Flask(__name__)
app.secret_key = "secret key"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST': # answer is submitted
        
        tweet = request.form['tweet']# request.form.GET.get('real_answer')
        real_answer = request.form['real_answer']# request.form.GET.get('real_answer')
        answer = request.form['answer'] # request.form.GET.get('answer')
        
        df = pd.DataFrame({'tweet': [tweet],
                           "answer": [answer]})

        if real_answer == answer:
            display_answer = "Correct!"
        else:
            display_answer = "Wrong!"
        flash(real_answer)
        flash(answer)
        return render_template("index.html", column_names=df.columns.values, row_data=list(df.values.tolist()),
                                link_column="answer",display_answer=display_answer, zip=zip)
    else:
        df_real = pd.read_csv("./src/static/tweet/actual.csv") 
        df_real["answer"] = "Real Tweet"
        df_fake = pd.read_csv("./src/static/tweet/generate.csv") 
        df_fake["answer"] = "Fake Tweet"

        df_real.rename(columns={ df_real.columns[0]: "tweet" }, inplace = True)
        df_fake.rename(columns={ df_fake.columns[0]: "tweet" }, inplace = True)

        df_combined = pd.concat([df_real, df_fake],ignore_index=True)
        df = df_combined.sample(n=1)

        return render_template("index.html", column_names=df.columns.values, row_data=list(df.values.tolist()),
                                link_column="answer",display_answer="", zip=zip)
        # return render_template("index.html")

@app.route('/documentation')
def documentation():
    return render_template("Documentation.html")

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        tweet_input = request.form['tweet_input']
        tweet_output =  run_gen(tweet_input)
        return render_template("tweet_form.html", tweet_input=tweet_input, tweet_output=tweet_output)
    else:
        return render_template("tweet_form.html")


@app.route("/get_tweet")
def get_tweet():
    df_real = pd.read_csv("./src/static/tweet/actual.csv") 
    df_real["answer"] = "real"
    df_fake = pd.read_csv("./src/static/tweet/generate.csv") 
    df_fake["answer"] = "fake"

    # df_real.rename(index={0: "tweet"})
    # df_fake.rename(index={0: "tweet"})
    df_real.rename(columns={ df_real.columns[0]: "tweet" }, inplace = True)
    df_fake.rename(columns={ df_fake.columns[0]: "tweet" }, inplace = True)

    df_combined = pd.concat([df_real, df_fake],ignore_index=True)
    df = df_combined.sample(n=1)

    # df = pd.DataFrame({'Patient Name': ["Some name", "Another name"],
    #                    "Patient ID": [123, 456],
    #                    "Misc Data Point": [8, 53]})

    return render_template("get_tweet.html", column_names=df.columns.values, row_data=list(df.values.tolist()),
                            link_column="answer", zip=zip)

@app.route("/get_answer", methods=['GET', 'POST'])
def get_answer():
    if request.method == 'POST':
        real_answer = request.form['real_answer']# request.form.GET.get('real_answer')
        answer = request.form['answer'] # request.form.GET.get('answer')
        if real_answer == answer:
            response = "correct"
        else:
            response = "wrong"
                
        flash(response)
        return render_template("get_answer.html")






if __name__ == "__main__":
    app.run(debug=True, port=8000)
    # Use gunicorn for serving in production
