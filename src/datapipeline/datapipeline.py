import pandas as pd
from sklearn.model_selection import train_test_split
import re


class Datapipeline():
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        
    def clean_text(self, text):
        text=str(text).lower() #lowercase
        text=re.sub('(https?:\/\/)(www\.)?\S+', '', text) #removes links http 
        text=re.sub('(pic\.)\S+','',text) #removes links to twitter pics/gifs
        text=re.sub(r'\@(\s)?\S+','@ ', text) #removes mentions
        text=re.sub(r'\#\S+','# ',text) #removes hashtags
        text=re.sub(r'[^\w\s]',' ',text)  #remove punctuation (adds a space)
        text=re.sub(r'\s+', ' ', text)   #removes doublespace
        return text
        
        
    def transform(self):
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df[(self.df['date'] > '2015-6-17') & (self.df['date'] <= '2020-6-17')] # last 5 years
        self.df['tweet_length'] = self.df['content'].apply(str)
        self.df['tweet_length'] = self.df['tweet_length'].apply(len)
        self.df = self.df[self.df.tweet_length > 20] # Remove 20 characters or less
        self.df['clean_text'] = self.df['content'].apply(self.clean_text)

    def split_data(self):
        '''

        :return: 2 single column dataframes
        '''
        train, val = train_test_split(self.df[['clean_text']], test_size=0.2, random_state=42)
        return train, val
