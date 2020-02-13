#!/usr/bin/env python
# encoding: utf-8

import tweepy  # https://github.com/tweepy/tweepy
import time, datetime
import pandas as pd
import numpy as np
import os
import argparse

# Twitter API credentials
consumer_key = 'w3516HrGStcxwn1YyBG5hVun1'
consumer_secret = '4pC5m7Q5ZQSsHZkX69Swujy0ORVlazyPApRrMSL7xwj4lzNWfn'
access_key = '1853768732-NCAu5j6CwsrCi4MYtupIULXW7gSWVCBTgrZOYdo'
access_secret = '2rKvB1sFIgRMC4rDWMcgiUjL04jlwn1NdkBpiYah6e8iR'

def process_df(df_temp):
    '''
    Filter on:
    1) English only
    2) Text contains $btc/$BTC
    3) Remove RT's
    4) Text doesn't contain more than one type of ticker symbol (if count $ unequal to count $btc).
    '''
    df_temp = df_temp[df_temp['lang'] == 'en']
    df_temp = df_temp[df_temp['text'].str.contains('\$btc', case=False)]
    df_temp = df_temp[~df_temp['text'].str.contains('^RT', case=False)]
    df_temp = df_temp[df_temp['text'].str.count('\$') == df_temp['text'].str.count('\$(B|b)(T|t)(C|c)')]
    df_temp['text'] = df_temp['text'].str.encode('utf-8')
    return df_temp

def get_all_tweets(screen_name):
    # Twitter only allows access to a users most recent 3240 tweets with this method

    # authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    # initialize a list to hold all the tweepy Tweets
    alltweets = []

    # make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name=screen_name, count=200)

    if len(new_tweets) == 0:
        return 0

    # save most recent tweets
    alltweets.extend(new_tweets)

    # save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    # keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        # all subsequent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name=screen_name, count=200, max_id=oldest)
        # save most recent tweets
        alltweets.extend(new_tweets)
        # update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

    # transform the tweepy tweets into a 2D array that will populate the csv.
    out_tweets = [[screen_name, tweet.id_str, tweet.created_at, tweet.text, tweet.lang,
                int(tweet.user.verified), tweet.user.followers_count, tweet.user.friends_count, tweet.user.listed_count,
                tweet.user.favourites_count, tweet.user.statuses_count, tweet.user.created_at,
                   tweet.user.id] for tweet in alltweets]
    df = pd.DataFrame(out_tweets, columns=['user', 'tweet_id', 'created_at', 'text', 'lang', 'verified',
                      'followers_count', 'friends_count', 'listed_count', 'favourites_count',
                                           'statuses_count', 'unix_months', 'twitter_user_id'])

    df['unix_months'] = df['unix_months'].apply(lambda x: round(time.mktime(x.timetuple()) / (60 * 60 * 24 * 30)))
    df = process_df(df)
    if len(df) > 0:
        with open('../data/user_files/{}_tweets.json'.format(screen_name), "w+") as output_file:
            output_file.write(df.to_json(orient='records', lines=True))
        return len(df)
    else:
        return 0

if __name__ == '__main__':
    '''
    Two use cases:
    1) generate list of users based on tweets (gen_list)
    2) generate processed username files consisting of tweets for that particular user.
    '''
    parser = argparse.ArgumentParser(description='Parse users')
    parser.add_argument('--gen_list', action="store_true", default=False, dest="gen_list")
    parser.add_argument('--file', action="store", dest="file")
    results = parser.parse_args()

    if results.gen_list:
        user_names = []
        files = os.listdir('../data/raw_tweets/')
        files = [x for x in files if '.json' in x]
        for filename in files:
            print('Processing {}'.format(filename))
            df = pd.read_json('../data/raw_tweets/{}'.format(filename), lines=True)
            df = process_df(df)['user'].to_dict()
            user_names.extend(list(set([df[i]['screen_name'] for i in df.keys()])))

        unique_names = np.array(list(set(user_names)))
        subsets_names = np.array_split(unique_names, 20)
        print('Gathered {} unique Twitter user names'.format(len(unique_names)))

        for i, sub in enumerate(subsets_names):
            np.save('../data/users/usernames_{}'.format(i), sub)
    else:
        total_tweets = 0
        tot_users_processed = 0
        skip = False
        for i in range(0, 20):
            file = base = '../data/users/usernames_{}.npy'.format(i)
            unique_names = np.load(file)
            print('------------------------------------')
            print('Downloading for file {}'.format(file))
            for name in unique_names:
                if (not skip) or (name == ''):
                    skip = False
                    print('Downloading data for user {}'.format(name))
                    finish_func = False
                    while not finish_func:
                        try:
                            tweets = get_all_tweets(name)
                            finish_func = True
                        except Exception as e:
                            if 'connection' in str(e).lower():
                                print(e)
                                time.sleep(30)
                            else:
                                print(e)
                                tweets = 0
                                finish_func = True
                    total_tweets += tweets
                    tot_users_processed += 1
                    print('Total amount of tweets {}, Total amount of users processed {}'.
                          format(total_tweets, tot_users_processed))
                    print('--------------------------------------------')