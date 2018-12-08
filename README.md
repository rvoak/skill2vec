# skill2vec

Word2vec models (Mikolov et al.) focus on the learned hidden representation of the input. In this code, we try to treat behaviour as a language. We train a skip-gram model using Gensim to learn vector representations of the skills. Then we reduce their dimensionality using t-SNE and plot them to observe the clustering of skills.
