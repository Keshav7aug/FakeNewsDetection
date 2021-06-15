from random import randint
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from newspaper import Article
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from nltk import pos_tag, word_tokenize
import pandas as pd
import traceback

from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
#print('//////////////////////////////////',os.getcwd())
from IsitFake.myData.tokLoader import tokenizer
from IsitFake.myData.modelLoader import model
WL = WordNetLemmatizer()
def open_file(name):
        df = pd.read_csv(f'Datasets/{name}.csv')
        return df
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'
def cleaner(data):
    #print(data)
    data = data.lower()
    data = re.sub('[^a-zA-Z ]','',data)
    token = data.split()
    #words = [WL.lemmatize(word) for word in token if not word in StopWords]
    words=[]
    for word,tag in pos_tag(token):
        wntag = get_wordnet_pos(tag)
        lemma = WL.lemmatize(word,wntag)
        words.append(lemma)
    data = ' '.join(words)
    return data
def Predictor(data):
    data = cleaner(data)
    sentence_length = 500
    #tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    data_tok = tokenizer.texts_to_sequences(data)
    data_tok = pad_sequences(data_tok,maxlen = sentence_length)
    ans=1
    #print('STARTTTTTTTTTTTTTTTTTTT')
    ans = model.predict(np.array(data_tok))[0][0]
    ans = 1 if ans > 0.5 else 0
    #print('ENDDDDDDDDDDDDDDDDDDDDDd')
    return ans
def news_article(url):
    if 'https://' not in url:
        return url
    print(f'HAHAHAHA {url} is a url')
    article = Article(url, language="en")
    article.download() 
    article.parse()
    return article.text
def extractor(queryDict):
    IsIt = ['Unreliable','Reliable']
    try:
        data = dict(queryDict)
        print('+++++++++++++++++++++++',data['data'])
        news = news_article(data['data'][0])
        ANS=Predictor(news)
        news.replace('\n','<br>')
        print("ANS==========",ANS)
        return news,IsIt[ANS]
    except Exception as E:
        print('boos',E)
        print(os.getcwd())
        print(traceback.format_exc())
        return '',''