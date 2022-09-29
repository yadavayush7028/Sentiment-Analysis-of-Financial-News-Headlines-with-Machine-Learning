import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import re

import nltk
from nltk.tokenize import RegexpTokenizer, TreebankWordTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import ngrams

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

data = pd.read_csv('./combined_data.csv')

# Data Cleaning and Prprocessing

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

# Function to convert to lower case, remove punctuations, links, digits, etc.

def clean(sentence):

  sentence=str(sentence)
  sentence = sentence.lower()
  sentence=sentence.replace('{html}',"") 
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', sentence)
  rem_url=re.sub(r'http\S+', '',cleantext)
  rem_url=re.sub(r'www\S+', '',cleantext)
  rem_num = re.sub('[0-9]+', '', rem_url)
  return rem_num

# Tokeninzer Function

def tknzr(sentence, rem_sw):
  filtered_text = clean(sentence)
  tokens = tokenizer.tokenize(filtered_text)
  if rem_sw == True:
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
  else:
    filtered_words = [w for w in tokens]
  return " ".join(filtered_words)

# Stemmer Function

def stmr(sentence, rem_sw):
  filtered_words = tknzr(sentence, rem_sw)
  filtered_words = filtered_words.split(" ")
  stem_words=[stemmer.stem(w) for w in filtered_words]
  return " ".join(stem_words)

# Lemmatizer Function

def lmtzr(sentence, rem_sw):
  filtered_words = tknzr(sentence, rem_sw)
  filtered_words = filtered_words.split(" ")
  lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
  return " ".join(lemma_words)

# Helper Function to apply above functions on whole dataframe.

def process_df(df, func, rem_sw=True):

    df['processed_text']=df['text'].map(lambda txt: func(txt, rem_sw))
    df.drop(['text'], axis=1, inplace=True)
    try:
      df.drop(['sntmt'], axis=1, inplace=True)
    except:
      print('')

    return df