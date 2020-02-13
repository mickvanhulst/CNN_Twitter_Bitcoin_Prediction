import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer

import string
import nltk
from nltk.tokenize import word_tokenize

def TFIDF(tweet_list, max_features=5000):
    '''
    Processes tweet list and finds top n keywords using TFIDF.
    Code from: https://shuaiw.github.io/2016/12/16/real-time-twitter-trend-discovery.html
    '''
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_vectorizer.fit_transform(tweet_list)
    return tfidf_vectorizer

def get_tfidf(text, tfidf_vectorizer):
    return list(tfidf_vectorizer.transform([text]).toarray())[0]

def get_features(tweets, tfidf_vectorizer, alphabet, pos_tags):
    features = []

    tokens = [word_tokenize(text) for text in tweets]
    tweet_bag = []
    tweet_bag.extend([item for sublist in tokens for item in sublist])
    d_pos = nltk.pos_tag(tweet_bag)
    tweets = ' '.join(tweets)

    TFIDF = get_tfidf(tweets, tfidf_vectorizer)
    features.append(sum(TFIDF))  # sum of tf*idf
    features.append(min(TFIDF))  # min of tf*idf
    features.append(max(TFIDF))  # max of tf*idf
    features.append(np.mean(TFIDF))  # mean of tf*idf
    features.append(np.var(TFIDF))  # variance of tf*idf

    character_counts = [0 for k in alphabet]
    for word in tweet_bag:
        for e, k in enumerate(alphabet):
            character_counts[e] += word.count(k)

    features += character_counts
    for pos_tag in pos_tags:
        count = 0
        for word in d_pos:
            if word[1] == pos_tag:
                count += 1
        features.append(count)


    return features

def gen_custom_features_tweet(tweets, tfidf_vectorizer):
    alphabet = string.ascii_letters + string.whitespace + string.punctuation
    pos_tags = ['CC', 'CD', 'DT', 'EX', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNS', 'PDT', 'POS', 'PRP',
                'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WRB']

    features = get_features(tweets, tfidf_vectorizer, alphabet, pos_tags)

    return np.array(features)