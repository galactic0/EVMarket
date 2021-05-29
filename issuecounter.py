# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 10:37:23 2020

@author: Aman
"""

import nltk

def countInstancefreq(data):
    words={}
    lines = [line for line in data["review_clean"]]
    is_noun = lambda pos: pos[:2] == 'NN'
    nouns_all=[]
    for text in lines:
        tokenized = nltk.word_tokenize(text)
        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
        nouns_all.append(nouns)

    for item in nouns_all:
        for word in item:
            if word not in words:
                words[word]=1
            elif word in words:
                words[word]+=1
            else:
                pass
    return words


# counting noun dictionary for +ve comments
