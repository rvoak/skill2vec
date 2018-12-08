
import pandas as pd
import numpy as np
import io
import requests
import gensim
import csv

url = "http://users.wpi.edu/~yutaowang/data/skill_builder_data.csv"
s = requests.get(url).content
df = pd.read_csv(io.StringIO(s.decode('ISO-8859-1')))

df2=df[['user_id','first_action','skill_id','ms_first_response','skill_name']]
df3=df2.sort_values(['user_id','ms_first_response'])



df3=df3.dropna()
ulist=df3['user_id'].unique()
df4=df3.set_index(['user_id'])
df4=df4.fillna(0)

vecs=[]




for i in ulist:

  x=df4.loc[i]
  a=x['skill_id']


  subvecs=[]
  df5=df4[df['user_id']==i]
  p=df5['skill_id'].values

  key=''
  for w in p:
    subvecs.append(str(w))


  vecs.append(subvecs)





model = gensim.models.Word2Vec(vecs, size=10, window=10, min_count=2, workers=10)
model.train(vecs,total_examples=len(vecs),epochs=10)


g=df.groupby(['skill_id','skill_name']).count()
g=g.reset_index()
g.columns
g2=g[['skill_id','skill_name']]
g2.set_index('skill_id')
print(g2.columns)
skillist={}

for k in model.wv.vocab:
  x=float(k)
  g3=g2[g2['skill_id']==x]
  t=g3['skill_name'].values[0]
  skillist[x]=t

with open('dict.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in skillist.items():
       writer.writerow([key, value])





for k in model.wv.vocab:
  t=float(k)
  print(skillist[t])

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# %matplotlib inline
def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(skillist[float(word)])

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

tsne_plot(model)
