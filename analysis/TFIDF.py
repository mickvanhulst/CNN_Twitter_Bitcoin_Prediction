import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer

def TFIDF(tweet_list, max_features=5000):
    '''
    Processes tweet list and finds top n keywords using TFIDF.
    Code from: https://shuaiw.github.io/2016/12/16/real-time-twitter-trend-discovery.html
    '''
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, min_df=5)
    tfidf_vectorizer.fit_transform(tweet_list)
    indices = np.argsort(tfidf_vectorizer.idf_)[::-1]
    features = tfidf_vectorizer.get_feature_names()
    return features, indices, tfidf_vectorizer.idf_

def main():
    df = pd.read_json('../data/processed/train_tweets.json', lines=True)

    # Find most occurring words using TFIDF
    features, indices, scores = TFIDF(df['text_kw'].tolist())
    keywords = {}
    for n in [250, 500, 1000]:
        keywords[n] = [[features[i], scores[i]] for i in indices[:n]]

    # Save words
    with open('../data/processed/keywords_test.json', "w+") as output_file:
        json.dump(keywords, output_file)

if __name__ == '__main__':
    main()