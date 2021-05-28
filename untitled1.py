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
pos_noun_dict=countInstancefreq(pos_data)
neg_noun_dict=countInstancefreq(neg_data)
neu_noun_dict=countInstancefreq(neu_data)

# to get the most likable, unlikable and neutral features of a service 
# we can find modes of the above sets


import plotly.figure_factory as ff


d=list(pos_noun_dict.keys())
group_labels=list(pos_noun_dict.keys())
fig = ff.create_distplot(d, group_labels , bin_size=0.2)
fig.show()
     
        
        
        
        


words_3={k:v for k, v in words.items() if v>15}

plt.bar(list(words_3.keys()), words_3.values(), color='r')
plt.show()    


import matplotlib.pyplot as plt


fig=plt.figure()
ax=plt.axes()

x = np.linspace(-1, 10, 1)
ax.plot(data["compound"]);
ax.show()

