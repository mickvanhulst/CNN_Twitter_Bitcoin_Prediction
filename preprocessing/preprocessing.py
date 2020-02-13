import numpy as np
import pandas as pd
import os
import re
import datetime
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string

def generate_tokens(text, stopwords, stemmer):
    ## Make word tokens out of both sets
    # Remove stop words
    word_tokens = word_tokenize(text)
    word_tokens = [word for word in word_tokens if word not in stopwords]
    return word_tokens

def remove_whitespaces(text):
    # Remove extra tabs/spaces/etc
    text = re.sub("\s\s+", " ", text)
    text = text.strip()
    return text

def replace_tokens(text, stopwords, stemmer, keyword=False):
    '''
    This function generates columns for three use cases:
    1) sentiment analysis using VADER (keyword = False and sentiment form = False), meaning
    basic data preprocessing.
    2) sentiment analysis using prior research, sentiment form = 1 and keyword = False, meaning adding
    basic preprocessing, adding NEGATION, QUESTION, EXCLAMATION and USERNAME tokens and tokenizing text.
    3) keyword analysis, sentiment form = False and keyword is true, meaning basic data preprocessing and
    tokenizing text.

    As the functions are used sequentially, I've made it so that sentiment form False and keyword false
    is processed first and the forms thereafter use that processed text to save computer power.
    '''
    ## General preprocessing
    # Remove unicode.
    text = str(text.encode('ascii', 'ignore').decode('ascii'))
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = text.replace('\n', ' ')

    # Remove characters that occur more than twice goooood, becomes good, but haappy stays haappy.
    text = re.sub(r'(.)\1+', r'\1\1', text)

    # Remove any digits that have to do with prices, so b4 won't be removed but 40 or 40,0 or 40.0 will be removed.
    text = re.sub('[0-9]+(.|,)[0-9]+', ' ', text)

    if keyword:
        # This is for keyword analysis and VADER sentiment analysis.
        # Remove URLs
        text = re.sub(r'(((https|http)?:\/\/)|www\.)(\w|\.|\/|\?|\=|\&|\%)*\b', ' ', text, flags=re.MULTILINE)

        # Remove mentions
        text = re.sub(r"@\w+", r" ", text)

        # Everything to lower case
        text = text.lower()
        # Remove all punctuations.
        punctuations = [x for x in string.punctuation]
        for p in punctuations:
            text = text.replace(p, '')

        # Remove stopwords.
        text = ' '.join(generate_tokens(text, stopwords, stemmer))

        return remove_whitespaces(text)

def generate_sentiment_vader(df, sid):
    '''
    https://www.nltk.org/_modules/nltk/sentiment/vader.html
    '''
    df['neg_score'] = df['text_vader'].apply(lambda x: sid.polarity_scores(x)['neg'])
    df['pos_score'] = df['text_vader'].apply(lambda x: sid.polarity_scores(x)['pos'])
    df['neu_score'] = df['text_vader'].apply(lambda x: sid.polarity_scores(x)['neu'])
    df['compound_score'] = df['text_vader'].apply(lambda x: sid.polarity_scores(x)['compound'])
    return df

def sentiment_PNT(df, pos_list, neg_list):
    '''
    Trading on Twitter paper
    Sentiment consists of:
    neg1 = N/T
    pos1 = (P-N)/(P+N)
    pos2 = log(1+P/1+N)

    We assign each of the three sentiments to the rows using a count of all words, based on the
    Harvard-IV dictionary. All negative words and postive words are tagged NEG and POS respectively.
    '''
    df['tokens'] = df['text_kw'].apply(lambda x: word_tokenize(x))
    df['harvard'] = df['tokens'].apply(
        lambda x: ['NEG' if word in neg_list else 'POS' if word in pos_list else 'NONE' for word in x])

    df['pos_cnt'] = df['harvard'].apply(lambda x: x.count('POS'))
    df['neg_cnt'] = df['harvard'].apply(lambda x: x.count('NEG'))
    df['tot_cnt'] = df['harvard'].apply(lambda x: len(x))

    df['neg1'] = df['neg_cnt'] / df['tot_cnt']
    df['pos1'] = (df['pos_cnt'] - df['neg_cnt']) / (df['pos_cnt'] + df['neg_cnt'])
    # Assume the paper describes log2
    df['pos2'] = np.log((1+df['pos_cnt']) / (1+df['neg_cnt']))

    # Fill NaN with zero.
    df = df.fillna(0)
    return df

def load_harvard_dict():
    # Load harvard dict, remove all hashtags and cast to lowercase.
    harvard_dict = pd.read_excel('../data/misc/harvard_pos_neg.xls', usecols=[0, 1, 2, 3], dtype={'Entry': np.string_})
    harvard_dict['Entry'] = harvard_dict['Entry'].apply(lambda x: re.sub(r'#.*', '', x.decode('utf-8')).lower())
    pos_list = harvard_dict['Entry'][harvard_dict['Positiv'] == 'Positiv'].tolist()
    neg_list = harvard_dict['Entry'][harvard_dict['Negativ'] == 'Negativ'].tolist()
    return pos_list, neg_list

def main():
    # get files to process
    REMOVE_BEFORE = '2017-06-01'
    TEST_DF_DATE = '2018-11-01'
    cnt_users = 0
    df = None
    stop = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    for file in os.listdir('../data/user_files'):
        cnt_users += 1
        temp_df = pd.read_json('../data/user_files/{}'.format(file), lines=True)
        temp_df = temp_df[(temp_df['created_at'].dt.date > np.datetime64(REMOVE_BEFORE))]
        if len(temp_df) == 0:
            continue
        temp_df['text_kw'] = temp_df['text'].apply(lambda x: replace_tokens(x, stop, stemmer, True))

        if cnt_users % 100 == 0:
            print('The total length of the DataFrame thus far is: {}, we have processed a total'
                  ' amount of {} users'.format(len(df), cnt_users))
        if df is None:
            df = temp_df
        else:
            df = pd.concat([df, temp_df], axis=0)

    print('The total length of the DataFrame is: {}, we have processed a total'
          'amount of {} users'.format(len(df), cnt_users))
    df['created_at'] = df['created_at'].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour))
    # Removed tweets that are before a certain date.
    df = df[(df['created_at'].dt.date >= np.datetime64(REMOVE_BEFORE))]# &
            # (df['created_at'].dt.date <= np.datetime64(MIN_DATE_TWEET))]

    test_df = df[df['created_at'] >= np.datetime64(TEST_DF_DATE)]
    train_df = df[df['created_at'] < np.datetime64(TEST_DF_DATE)]

    # Filter, make sure that all users in test are also in train and vice versa.
    print(len(test_df), len(train_df))
    train_df = train_df[train_df['user'].isin(test_df['user'].unique())]
    test_df = test_df[test_df['user'].isin(train_df['user'].unique())]
    print(len(test_df), len(train_df))

    print('Length of test df {}, length of train df {}, length of entire df {}, amount of users in test df {}'
          .format(len(test_df), len(train_df), len(df), len(test_df['user'].unique())))
    with open('../data/processed/test_tweets.json', "w+") as output_file:
        output_file.write(test_df.to_json(orient='records', lines=True))

    with open('../data/processed/train_tweets.json', "w+") as output_file:
        output_file.write(train_df.to_json(orient='records', lines=True))
if __name__ == '__main__':
    main()