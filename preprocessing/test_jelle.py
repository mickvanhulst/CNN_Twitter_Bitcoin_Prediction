import tweepy  # https://github.com/tweepy/tweepy

# Twitter API credentials
consumer_key = ''
consumer_secret = ''
access_key = ''
access_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

tweet_ids = [100203467251007488]
for tweet_id in tweet_ids:
    try:
        tweet = api.get_status(tweet_id)
        print(tweet.text)
    except:
        print('Tweet with Id {} does not exist'.format(tweet_id))
        continue