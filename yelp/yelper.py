#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 21:30:29 2020

@author: matthewmckenna
"""

import inspect
import nltk
import json
import pandas as pd
import yelp
from yelp.client import Client
import requests


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    #print("{:-<40} {}".format(sentence, str(score)))
    
    
client = Client('ouKQYH4SYmRdoKUXxvpOshjPMNMUqEqKB97KE95gg-9B1-zy7eRdrZypN-cWYIkJlxfLeIWVcCIZEJPGMdgcAvPivLtACvuvg2mN7NWZBdcMPTdbcAVyCvMEdnPwXXYx')

#business_response = client.business.get_by_id('yelp-san-francisco')


api_key='ouKQYH4SYmRdoKUXxvpOshjPMNMUqEqKB97KE95gg-9B1-zy7eRdrZypN-cWYIkJlxfLeIWVcCIZEJPGMdgcAvPivLtACvuvg2mN7NWZBdcMPTdbcAVyCvMEdnPwXXYx'
headers = {'Authorization': 'Bearer %s' % api_key}

def get_business_id(term, location):
    
    url='https://api.yelp.com/v3/businesses/search'
    
    # In the dictionary, term can take values like food, cafes or businesses like McDonalds
    params = {'term':term,'location':location}

    
    req=requests.get(url, params=params, headers=headers)
     
    
    json.loads(req.text)
    
    parsed = json.loads(req.text)
    #print(json.dumps(parsed, indent=4))
    
    
    business = parsed["businesses"] 
    #print(business)
    
    '''
    for business in businesses[0:1]:
        print("Name:", business["name"])
        print("Rating:", business["rating"])
        print("Address:", " ".join(business["location"]["display_address"]))
        print("Phone:", business["phone"])
        print("ID:", business["id"])
        print("\n")
    '''
    
    id = business[0]["id"]
    
    return(id)
    
    
def get_reviews(id): 

    # In the dictionary, term can take values like food, cafes or businesses like McDonalds
    
    url='https://api.yelp.com/v3/businesses/' + id + '/reviews'
    params = {'locale':'en_US'}
    
    
    req=requests.get(url, params=params, headers=headers)
    
    parsed = json.loads(req.text)
    revs = parsed["reviews"]
    out_revs = []
    
    out_df = pd.DataFrame()
    
    for rev in revs:
        
        tokenized_sentence = nltk.word_tokenize(rev['text'])
        sentence = rev['text']
        out_revs.append(str(rev['text']))
                
        sid = SentimentIntensityAnalyzer(lexicon_file='vader_lexicon.txt')
        pos_word_list=[]
        neu_word_list=[]
        neg_word_list=[]
        
        for word in tokenized_sentence:
            if (sid.polarity_scores(word)['compound']) >= 0.1:
                pos_word_list.append(word)
            elif (sid.polarity_scores(word)['compound']) <= -0.1:
                neg_word_list.append(word)
            else:
                neu_word_list.append(word)                
        
        score = sid.polarity_scores(sentence)
        out_df = out_df.append(pd.DataFrame({'pos':','.join(pos_word_list), 'neg':','.join(neg_word_list), 'score':[score['compound']], 'rev':rev['text'], 'rating':rev['rating']}))
        
    return(out_df)


def main(in_search, in_loc): 

    id_= get_business_id(in_search, in_loc)
    revs = get_reviews(id_)
   
    return(revs)


