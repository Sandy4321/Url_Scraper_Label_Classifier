# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
from lxml import etree as et
import requests
import re
import numpy as np
import pandas as pd
from htmllaundry import strip_markup
from multiprocessing import Pool
import unicodedata

import nltk
import itertools
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.classify import NaiveBayesClassifier
from nltk.stem.snowball import FrenchStemmer
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import WordPunctTokenizer
import nltk.classify.util, nltk.metrics
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB

import xgboost as xgb
import operator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import sklearn.metrics

def isNaN(num):
    return num != num

# fr_urls = pd.read_csv('france_urls.csv', header=None)
# print fr_urls.head()
#
# length = len(fr_urls)
# n = 50
#
# print 'length'
# print n
#
# print
# print "Scraping the url list"
#
#
# fr_urls = pd.DataFrame(fr_urls)
# fr_urls[0] = fr_urls[0].str.replace('(/\*)','/')
#
# df = pd.DataFrame("None", index=range(0,n), columns=['url', 'title', 'body', 'description'])
#
#
# for i in range(0,n):
#
#     url = fr_urls.loc[i,0]
#     if "http://" not in url:
#         url = "http://" + url
#
#     # if "/*" in url:
#
#
#     print url
#
#     try:
#
#         response = requests.get(url)
#         txt = response.text
#
#         tree = et.HTML(txt)
#
#         soup = BeautifulSoup(txt, 'lxml')
#
#         body_1 = soup.body('p')
#         body = unicode.join(u'\n',map(unicode,body_1))
#         # body = unicodedata.normalize('NFKD', body_1).encode('ASCII','ignore')
#         body = strip_markup(" ".join(body.split()))
#
#         df.loc[i,'body'] = body
#
#     except:
#         pass
#
#     try:
#         metakeywords = tree.xpath( "//meta" )
#     except:
#         metakeywords = "None"
#
#     # df.loc[i,'keywords'] = "None"
#     #
#     # df.loc[i,'description'] = "None"
#     for m in metakeywords:
#         if m.get('name') and 'title' in m.get('name'):
#         # print metakeywords[1].get('http-equiv')
#             df.loc[i,'title'] = m.get('content')
#             break
#         elif m.get('property') and 'title' in m.get('property'):
#             df.loc[i,'title'] = m.get('content')
#             break
#         else:
#             pass
#
#     keywords = 'None'
#     for m in metakeywords:
#         if m.get('http-equiv') and 'keywords' in m.get('http-equiv'):
#         # print metakeywords[1].get('http-equiv')
#             df.loc[i,'keywords'] = m.get('content')
#             break
#         elif m.get('name') and 'keywords' in m.get('name'):
#             df.loc[i,'keywords'] = m.get('content')
#             break
#         else:
#             pass
#
#     description = 'None'
#     for m in metakeywords:
#         if m.get('name') and 'description' in m.get('name'):
#         # print metakeywords[1].get('http-equiv')
#             df.loc[i,'description'] = m.get('content')
#             break
#         elif m.get('property') and 'description' in m.get('property'):
#             df.loc[i,'description'] = m.get('content')
#             break
#         else:
#             pass
#
#     df.loc[i,'url'] = url
#
# print df.head()
#
# # f_out = "france_urls_scraped_2.csv"
# # df.to_csv(f_out, index=0, encoding='utf-8')
#
# print "Number of failed scrapes"
# print len(df[df['body'] == ''])
# print
#
# # print "{0} lines written to file {1}".format(len(df),f_out)


####
####
####
#### Combining scrapes with labels

# print
# print "Joining scraped data to labels"
#
#
# fname = "france_urls_scraped.csv"
# df_train = pd.read_csv(fname, index_col=None)
#
# dfile = pd.read_csv('domain_list_1.csv', index_col=None)
#
# dfile = dfile[['publisher_url', 'Segment1']]
#
# dfile['publisher_url'] = dfile['publisher_url'].apply(lambda x: x.replace('.*',""))
# dfile['publisher_url'] = dfile['publisher_url'].str.replace('(http://www.)','')
#
# print dfile.head(10)
#
# # print arr[1]
#
#
# df_train = df_train[df_train['body'] != '']
#
#
# df_train.loc[isNaN(df_train['keywords']), 'keywords'] = 'None'
#
#
# print df_train.head(10)
# df_train = df_train.to_string

# uni_1 = lambda x: x.decode('utf-8','ignore')
# uni_2 = lambda x: unicodedata.normalize('NFKD', x).encode('ASCII','ignore')
#
# # df_train['keywords'] = df_train['keywords'].apply(uni)
# # df_train['description'] = df_train['description'].apply(uni_1)
# # df_train['title'] = df_train['title'].apply(uni_1)
# # df_train['body'] = df_train['body'].apply(uni_1)
#
# df_train['description'] = df_train['description'].apply(uni_2)
# df_train['title'] = df_train['title'].apply(uni_2)
# df_train['body'] = df_train['body'].apply(uni_2)


# df_train['title'] = df_train['keywords'] + df_train['description'] + df_train['title']
#
# df_train.drop(['keywords','description'], axis=1, inplace=True)
#
# df_train['title'] = df_train['title'].str.lower()
#
# parse = lambda x: urlparse.urlsplit(x).netloc
#
# df_train['url'] = df_train['url'].str.replace('(http://)','')
# df_train['url'] = df_train['url'].str.replace('(www.)','')
#
#
# print df_train.head(20)
#
# df_joined = pd.merge(left=dfile, right=df_train, how='inner', left_on = 'publisher_url', right_on = 'url')
#
# df_joined=df_joined.rename(columns = {'Segment1':'label'})
# df_joined.drop(['publisher_url'], inplace = True, axis=1)
# #
# print df_joined['label'].value_counts()
#
# print df_joined.head()


####
####
####
#### XG Boost model

print
print "Training xg-boost model"

# df_train = pd.read_csv(fname, index_col=0)
df_train = pd.read_csv("france_urls_labeled.csv", index_col=None)

df_train = df_train.loc[1:500,:]

val_urls = pd.read_csv("test_urls_scraped.csv", index_col=None)

print df_train.head()
print val_urls.head()

# df_val = pd.read_csv(fname, index_col=0)

# df_train = pd.concat([df_train, df_train_2], axis=0)

print df_train['label'].value_counts()

# print df['keywords'][12:15]
# print isNaN(df['keywords'][2])

stops = set(stopwords.words("french"))

# print stops[1:10]
stop = lambda x: [w for w in x if w not in stops]

empty = lambda x: [w for w in x if not w is '']

join_all = lambda x: " ".join(x)

nan_cleaner = lambda x: ["None" if isNaN(x) else x]

ls = FrenchStemmer()

tokenizer = WordPunctTokenizer()

stem = lambda x: [ls.stem(w) for w in x]

u_1 = lambda x: x.decode('utf-8')
unic = lambda x: unicodedata.normalize('NFKD', x).encode('ascii','ignore')


def processing(df):

    df.loc[isNaN(df['keywords']),:]  = 'None'
    df.loc[isNaN(df['body']),:]  = 'None'
    df.loc[isNaN(df['description']),:]  = 'None'
    df.loc[isNaN(df['title']),:]  = 'None'

    df['keywords'] = df['keywords'].str.replace(";"," ")

    df['body'] = df['title'] + df['body'] + df['description'] + df['keywords']

    df['body'] = df['body'].apply(lambda x: re.sub(r' \d+','NUM',str(x)))

    df['body'] = df['body'].apply(u_1)
    # df['body'] = df_train['body'].apply(unic)

    # df['keywords'] = df['keywords'].apply(lambda x: " ".join(x))
    #
    # df['body'] = df[['body','keywords']].apply(lambda x: ' '.join(x) if len(x) != 0 else x, axis=1)

    df['body'] = df['body'].apply(lambda x: BeautifulSoup(x, "html.parser"))

    df['body'] = df['body'].apply(lambda x: re.sub('[^A-Za-z0-9_]',' ',str(x)))

    df['body'] = df['body'].str.lower()


    df['body'] = df['body'].apply(lambda x: tokenizer.tokenize(x))

    df['body'] = map(stop, df['body'])

    df['body'] = map(empty, df['body'])

    df['body'] = map(stem, df['body'])

    df['body'] = map(join_all, df['body'])

    return df


df_train = processing(df_train)
df_val = processing(val_urls)

# print df_train['body'][12]

train_1 = df_train.groupby("label").filter(lambda x: len(x) <= 25)

train_2 = df_train.groupby("label").filter(lambda x: len(x) > 25)

train_1 = train_1.reset_index(drop=True)

train_1['label'] = 'others2'

# df_train = train_2
df_train = train_2[(train_2['label'] != 'Actu media') & (train_2['label'] != 'Others')]
# df_train = train_2.drop(['Actu media', 'Others'],axis=1)

print df_train.head()

# df_train = df_train.loc[(df_train['label']!='Other'),:]

df_lab = pd.DataFrame(df_train['label'].value_counts(), index=None)
df_lab.reset_index(drop=False, inplace=True)

num_class = len(df_train['label'].value_counts())

df_train['label'] = df_train['label'].astype('category')
df_train['label'] = df_train['label'].cat.codes


# vectoriser = CountVectorizer(analyzer='word', tokenizer=None, preprocessor=None, stop_words=None, max_features=2000)

vectoriser = CountVectorizer(ngram_range=(1, 3), token_pattern=r"\b\w+\b", min_df=1, max_features=2000)
# tokenizer = TreebankWordTokenizer()

train_data_features = vectoriser.fit_transform(df_train['body'])
val_data_features = vectoriser.fit_transform(df_val['body'])

train_data_features = train_data_features.toarray()
val_data_features = val_data_features.toarray()

print train_data_features.shape

print val_data_features.shape

vocab = vectoriser.get_feature_names()

dist = np.sum(train_data_features, axis=0)


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

print vocab[:100]
create_feature_map(vocab)

X_val = val_data_features

X_train, X_test, y_train, y_test = train_test_split(train_data_features, df_train['label'], test_size=0.30, random_state=49)

# X_test, X_val, y_test, y_val = train_test_split(train_data_features, df_train['label'], test_size=0.30, random_state=39)

# params = {'objective':'multi:softmax', 'num_class': num_class, 'max_depth': 2, 'eta': 0.1}
#
# T_train = xgb.DMatrix(X_train,y_train)
# X_test_xgb = xgb.DMatrix(X_test)
#
# gbm = xgb.train(params, T_train, 200)
# Y_pred = gbm.predict(X_test_xgb)
#
# pred= Y_pred
nc = len(df_train['label'].value_counts())

clf = xgb.XGBClassifier(max_depth = 2, n_estimators=500, learning_rate=0.1, nthread=4, objective='multi:softmax', seed=4282)

clf.fit(X_train, y_train, early_stopping_rounds=20, eval_metric="mlogloss", eval_set=[(X_test, y_test)])

pred = clf.predict(X_test)

pred_val = clf.predict(X_val)





# print Y_pred
lab = list(df_train['label'].value_counts().index)

df_lab['codes'] = list(df_train['label'].value_counts().index)

df_lab=df_lab.rename(columns = {'index':'labels', 'label':'count'})


print df_lab

print

print df_train.head()


print 'validation matrix'
print confusion_matrix(y_test, pred, labels=lab)


print sklearn.metrics.f1_score(y_test, pred, labels=lab, average='micro')


val_urls['pred_label'] = pred_val
val_urls = val_urls[['url', 'pred_label']]

f_out = 'test_labeled.csv'
val_urls.to_csv(f_out)
print "{0} lines written to file {1}".format(len(val_urls),f_out)
