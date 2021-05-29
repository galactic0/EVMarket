# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 05:41:34 2020

@author: Aman
"""

import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from issuecounter import countInstancefreq

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color = 'white',
        max_words = 200,
        max_font_size = 40, 
        scale = 3,
        random_state = 42
    ).generate(str(data))

    fig = plt.figure(1, figsize = (20, 20))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(wordcloud)
    plt.show()




def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    

def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)




data=pd.read_csv('all comp review da.csv')
data=data.drop(data.columns[1], axis=1, inplace=False)
data.describe()
# clean text data
data["review_clean"] = data["Review"].apply(lambda x: clean_text(x))
# add sentiment anaylsis columns
sid = SentimentIntensityAnalyzer()
data["sentiments"] = data["review_clean"].apply(lambda x: sid.polarity_scores(x))
#turning sentiments dict to separate columsn for ease of access and peeking
data = pd.concat([data.drop(['sentiments'], axis=1), data['sentiments'].apply(pd.Series)], axis=1)
# add number of characters column
data["nb_chars"] = data["Review"].apply(lambda x: len(x))
# add number of words column
data["nb_words"] = data["Review"].apply(lambda x: len(x.split(" ")))




# highest positive sentiment reviews (with more than 5 words)
data[data["nb_words"] >= 5].sort_values("pos", ascending = False)[["Review", "pos"]].head(10)
#The main the thing where we get the reviews from
neg_data=data[data["nb_words"] >= 3]
neg_data=neg_data[neg_data["compound"]<0].sort_index(ascending=True)
pos_data=data[data["nb_words"] >= 3]
pos_data=pos_data[pos_data["compound"] >0.25]
pos_data=pos_data.sort_index(ascending=True)


# extracting neutral data where with min. 3 words 
neu_data=data[data["nb_words"] >= 3]
neu_data=neu_data[neu_data["compound"]>=0]
neu_data=neu_data[neu_data["compound"]<=0.25].sort_index(ascending=True)


cat_index=list(neu_data.index)+list(pos_data.index)+list(neg_data.index)

data_index=[ x for x in range(0,715)]

uncat_index = [ x for x in data_index if x not in cat_index]

uncat_data=data.iloc[uncat_index,:]

uncat_data=uncat_data.loc[:,["company","Review","review_clean", "pos","compound"]]




pos_data.groupby(["company"]).count()

neg_data.groupby(["company"]).count()

#neutral
neu_data=data[data["nb_words"] >= 5].sort_values("neu", ascending = False)[["Review", "neu", "review_clean"]]
neu_data=neu_data[neu_data["neu"]> 1]


# show word clouds for pos, neg and neutral comments
show_wordcloud(pos_data["review_clean"])
show_wordcloud(neg_data["review_clean"])
show_wordcloud(neu_data["review_clean"])

#counting issue frequency in pos, neg and neu comments
pos_issue_count=countInstancefreq(pos_data)
neg_issue_count=countInstancefreq(neg_data)
neu_issue_count=countInstancefreq(neu_data)

posdf=pd.DataFrame.from_dict({'pos': list(pos_issue_count.keys()), 'count':list(pos_issue_count.values()) })
negdf=pd.DataFrame.from_dict({'neg':list(neg_issue_count.keys()),
                              'count':list(neg_issue_count.values())
                              })
neudf=pd.DataFrame.from_dict({'neu': list(neu_issue_count.keys()),
                              'count': list(neu_issue_count.values())
                              })                                         





pos_fig = px.bar(posdf[posdf['count']>18], x='pos', y='count')
neg_fig = px.bar(negdf[negdf['count']>18], x='neg', y='count')
neu_fig = px.bar(neudf[neudf['count']>18], x='neu', y='count')

pos_fig.show()
neu_fig.show()
neg_fig.show()
  
