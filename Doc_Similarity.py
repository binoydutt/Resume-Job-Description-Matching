# -*- coding: utf-8 -*-
"""
Created on Sun May 07 02:20:45 2017

@author: binoy
"""

import pandas as pd
from gensim.summarization import keywords
import matplotlib.pyplot as plt
df =pd.read_csv('data.csv')
jd = df['Job Description'].tolist()
companies = df['company'].tolist()
positions = df['position'].tolist()

from gensim import models
docs = []
for i in range(len(jd)):
    sent = models.doc2vec.LabeledSentence(words = jd[i].split(),tags = ['{}_{}'.format(companies[i], i)])
    #sent = models.doc2vec.LabeledSentence(words = jd[i].split(),tags = ['jd'])
    docs.append(sent)
    
model = models.Doc2Vec(alpha=.025, min_alpha=.025, min_count=1)
model.build_vocab(docs)

for epoch in range(10):
    model.train(docs)
    model.alpha -= 0.002  # decrease the learning rate`
    model.min_alpha = model.alpha  # fix the learning rate, no decay
    
with open('resumeconverted.txt','r') as f:
    resume = f.read()

    
from sklearn.manifold import MDS
data = []
for i in range(len(jd)):
    data.append(model.docvecs[i])

data.append(model.infer_vector(resume))

mds = MDS(n_components=2, random_state=1)
pos = mds.fit_transform(data)
#print pos
xs,ys = pos[:,0], pos[:,1]
for x, y in zip(xs, ys):
    plt.scatter(x, y)
#    plt.text(x, y, name)
xs2,ys2 = xs[-1], ys[-1]
plt.scatter(xs2, ys2, c='Red', marker='+')
plt.text(xs2,ys2,'resume')
plt.savefig('distance.png')
plt.show()


from sklearn.metrics.pairwise import cosine_distances
cos_dist =[]
for i in range(len(data)-1):
    print i
    cos_dist.append(float(cosine_distances(model.infer_vector(resume),data[i])))
    

key_list =[]

for j in jd:
    key =''
    for word in keywords(j).split('\n'):
        key += '{} '.format(word)
    key_list.append(key)

summary = pd.DataFrame({
        'Company': companies,
        'Postition': positions,
        'Cosine Distances': cos_dist,
        'Keywords': key_list,
        'Job Description': jd
    })
z =summary.sort('Cosine Distances', ascending=False)
z.to_csv('Summary.csv',encoding="utf-8")