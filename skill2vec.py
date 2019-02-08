
import pandas as pd
import numpy as np
import io
import requests
import gensim
import csv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def read_data():
  url = "http://users.wpi.edu/~yutaowang/data/skill_builder_data.csv"
  s = requests.get(url).content
  df = pd.read_csv(io.StringIO(s.decode('ISO-8859-1')))
  return df


def clean_data(df):
  df2=df[['user_id','first_action','skill_id','ms_first_response','skill_name']]
  df3=df2.sort_values(['user_id','ms_first_response'])
  df3=df3.dropna()
  ulist=df3['user_id'].unique()
  df4=df3.set_index(['user_id'])
  df4=df4.fillna(0)
  return df4


def get_student_vectors(df4,df):
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
   
  return vecs

def build_w2v_model(vecs):
  model = gensim.models.Word2Vec(vecs, size=10, window=10, min_count=2, workers=10)
  model.train(vecs,total_examples=len(vecs),epochs=10)
  return model


def build_skill_dict(df,model):
  
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
          
  return skillist

def tsne_plot(model,skillist):
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

def main():
  df=read_data()
  df4=clean_data(df)
  vecs=get_student_vectors(df4,df)
  model=build_w2v_model(vecs)
  skillist=build_skill_dict(df,model)
  tsne_plot(model,skillist)
  
if __name__=='main':
  main()
