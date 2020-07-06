
import tweepy
import pandas as pd
from textblob import TextBlob

#import matplotlib.pyplot as plt
#%matplotlib inline

ckey = 
csecret = 
atoken = 
asecret = 

auth = tweepy.OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

api = tweepy.API(auth)


def get_sentiment(in_str):
    
    sent = TextBlob(in_str).sentiment.polarity
    if sent>.5: 
        sentiment='great'
        
    elif sent>=0: 
        sentiment='neutral'
    elif sent>-.5: 
        sentiment='bad'
    else: 
        sentiment='terrible'
    
    
    return (sentiment, sent)


def get_all_tweets(screen_name):
#Twitter only allows access to a users most recent 3240 tweets with this method

        #authorize twitter, initialize tweepy
        auth = tweepy.OAuthHandler(ckey, csecret)
        auth.set_access_token(atoken, asecret)
        api = tweepy.API(auth)

        #initialize a list to hold all the tweepy Tweets
        alltweets = []

        #make initial request for most recent tweets (200 is the maximum allowed count)
        new_tweets = api.user_timeline(screen_name = screen_name,count=200, include_rts = False, tweet_mode="extended")

        #save most recent tweets
        alltweets.extend(new_tweets)

        #save the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        #keep grabbing tweets until there are no tweets left to grab
        while len(new_tweets) > 0:

                #all subsiquent requests use the max_id param to prevent duplicates
                new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest, include_rts = False, tweet_mode="extended")

                #save most recent tweets
                alltweets.extend(new_tweets)

                #update the id of the oldest tweet less one
                oldest = alltweets[-1].id - 1
        outtweets = [[tweet.id_str, tweet.created_at, tweet.full_text] for tweet in alltweets]

        return(outtweets)
       
   

def _removeNonAscii(s): return "".join(i for i in s if ord(i)<128)


def main(in_name):
    
    ret = get_all_tweets(in_name)

    tweets = [_removeNonAscii(tweet[2]) for tweet in ret]
    dates = [tweet[1] for tweet in ret]
    senti = [get_sentiment(x)[1] for x in tweets]
    
    df = pd.DataFrame({'dates':dates, 's':senti})
    df.index=dates
    df.drop('dates', 1,inplace=True)
    
    #df[["s"]].resample("3d").mean().plot(figsize=(15,4))
    
    #df.plot(figsize=(20,10), linewidth=5, fontsize=20)
     
    #return(zip(tweets,senti))
    return((tweets, dates, senti))

