# -*- coding: utf-8 -*-
'''
0. Data: I Sad (5566)
1. TF-IDF Vectorize --> KMeans Cluster
2. Word2Vec Vectorize --> KNN Classifier
'''

# --- import 
import pandas as pd
import numpy as np

import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# --- data
lyrics = '''那一年默默無言 只能選擇離開
無邪的笑容已經 不再精彩
你害怕結局所以 拼命傷害
說是我擋住你的 美好未來

你堅決 不希望我等待 我便默默的讓你走開
如今你 受了傷回來 叫我如何接受這安排

我難過的是 放棄你 放棄愛 放棄的夢被打碎 忍住悲哀
我以為 是成全 你卻說你更不愉快

我難過的是 忘了你 忘了愛 盡全力忘記我們 真心相愛
也忘了告訴你 失去的不能重來

那一年默默無言 只能選擇離開
無邪的笑容已經 不再精彩
你害怕結局所以 拼命傷害
說是我擋住你的 美好未來

更多更詳盡歌詞 在 ※ Mojim.com　魔鏡歌詞網 
你堅決 不希望我等待 我便默默的讓你走開
如今你 受了傷回來 叫我如何接受這安排

我難過的是 放棄你 放棄愛 放棄的夢被打碎 忍住悲哀
我以為 是成全 你卻說你更不愉快

我難過的是 忘了你 忘了愛 盡全力忘記我們 真心相愛
也忘了告訴你 失去的不能重來

我難過的是 放棄你 放棄愛 放棄的夢被打碎 忍住悲袞
我以為 是成全 你卻說你更不愉快

我難過的是 忘了你 忘了愛 盡全力忘記我們 真心相愛
也忘了告訴你 失去的不能重來

我難過的是 放棄你 放棄愛 放棄的夢被打碎 忍住悲哀
我以為 是成全 你卻說你更不愉快

我難過的是 忘了你 忘了愛 盡全力忘記我們 真心相愛
也忘了告訴你 失去的不能重來'''

docs = lyrics.split('\n\n')
for i, _ in enumerate(docs):
    print(i+1)
    print(_)

punct = list(u'''\n +-％%:!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳丨﹐､﹒﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠々‖•·ˇˉ―′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻︽︿﹁﹃﹙﹛﹝（｛“‘—_…~/#><''')
    
# save to dataframe
df0 = pd.DataFrame({'doc':docs}) 

# add a column "words" for list of words, using jeiba
df0['words'] = df0['doc'].map(lambda x: [_ for _ in jieba.cut(x) if _ not in punct]) 

# --- TF-IDF Vectorize and KMeans Cluster
# transformwords to vecters, using sklearn.feature_extraction.text.CountVectorizer
df0['words_str'] = df0['words'].map(lambda x: ' '.join(x))

# CountVectorizer: --> vecs1
cv = CountVectorizer()
vecs1 = cv.fit_transform(df0['words_str']).toarray()


# TfidfTransformer: --> vecs2
tfidf = TfidfTransformer()
vecs2 = tfidf.fit_transform(vecs1).toarray()
df0['tfidf_vector'] = [list(_) for _ in vecs2]

# kmeans
X = np.array(list(df0['tfidf_vector'].values))
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
df0['kmeans_group'] = kmeans.predict(X)
df0[['kmeans_group', 'doc']]

# show groups
df0['left_100'] = df0['doc'].map(lambda x: x[:100])
df0.groupby(['kmeans_group', 'left_100']).size().reset_index()


# --- Word2Vec Vectorize and KNN Classifier
# W2V
w2v = Word2Vec(df0['words'], min_count=2, size=10)


def words_to_w2v_avg(w2v, words):
    vecs = np.array([w2v[_] for _ in words if _ in w2v]) # get word's vector in word in w2v's vocabulary
    return vecs.mean(axis=0) # return average of vectors

df0['w2v_vector_avg'] = df0['words'].map(lambda x: words_to_w2v_avg(w2v, x))
df0[['doc', 'w2v_vector_avg']]

# --- W2C + K-means 
X2= np.array(list(df0['w2v_vector_avg'].values))
w2vkmeans = KMeans(n_clusters=4, random_state=0).fit(X2)
df0['w2v_kmeans_group'] = w2vkmeans.predict(X2)

# --- KNN classifier
# train KNN model
X = np.array(list(df0['w2v_vector_avg'].values))
knn = NearestNeighbors(n_neighbors=10).fit(X)

# predict
dists, ixs = knn.kneighbors(df0.loc[1, 'w2v_vector_avg'])

# print 3nn distance and doc
for dist, ix in zip(dists[0], ixs[0]):
    print('%s\n%s\n'%(dist, df0.loc[ix, 'doc']))

#cosine similarity matrix    
dfX = pd.DataFrame() 
dfX2 = pd.DataFrame() 
for ix, row in df0.iterrows():
    dfX[ix]=df0['w2v_vector_avg'].map(lambda vec: (round(cosine_similarity(vec, row['w2v_vector_avg'])[0,0],4)))
    dfX2[ix]=dfX[ix].map(lambda x: 1 if x>0.695 else 0)

dfX2[ix]=dfX[ix].map(lambda x: 1 if x<=0.3258 else 0)
    
dfX.sum()
    
(dfX.astype(float)>0).sum().sum()


    
dfX2.to_csv(u'../data/cosine_similarity_matrix.csv',sep=';')


test1=cosine_similarity([1,1],[-1,-1])[0][0]
test1
