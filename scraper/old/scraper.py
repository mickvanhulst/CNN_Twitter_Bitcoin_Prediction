import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import datetime
import time
 
consumer_key = 'w3516HrGStcxwn1YyBG5hVun1'
consumer_secret =  '4pC5m7Q5ZQSsHZkX69Swujy0ORVlazyPApRrMSL7xwj4lzNWfn'
access_token = '1853768732-NCAu5j6CwsrCi4MYtupIULXW7gSWVCBTgrZOYdo'
access_secret = '2rKvB1sFIgRMC4rDWMcgiUjL04jlwn1NdkBpiYah6e8iR'
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth)


class MyListener(StreamListener):
 
    def on_data(self, data):
        try:
            i = datetime.datetime.now()
            with open('../data/tweets_%s_%s_%s_%s.json' % (keyword, i.day, i.month, i.year), 'a') as f:
                f.write(data)
                print('Scraped row, yay @{}:)!'.format(time.strftime('%X')))
                return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True
 
    def on_error(self, status):
        print(status)
        return True
keyword = '$btc'
twitter_stream = Stream(auth, MyListener())
twitter_stream.filter(track=[keyword])
