import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import cluster
import numpy as np
import os
import gensim
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

def elbow_method(X):
    # k means determine k
    distortions = []
    K = range(1, 100)
    for k in K:
        print('processed {}'.format(k))
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

def main():
    word2vec_train = False
    model_file = '../data/models/word2vec.model'
    if not word2vec_train:
        df = pd.read_json('../data/processed/train_tweets_stem.json', lines=True)
        sentences = df['text_kw'].apply(lambda x: word_tokenize(x)).tolist()

        model = gensim.models.Word2Vec(sentences, min_count=1)#, size=100)
        model.train(sentences, total_examples=len(sentences), epochs=15)
        model.save(model_file)
    else:
        model = gensim.models.Word2Vec.load('../data/models/word2vec.model')
    X = model[model.wv.vocab]

    # Set K manually based on elbow method.
    K = 25
    if K is None:
        elbow_method(X)
    else:
        print('Creating centroids for K equals {}'.format(K))
        # Cluster based on similarities (determining n clusters for K-means is quite hard, perhaps there's an alternative).
        kmeans = cluster.KMeans(n_clusters=K)
        kmeans.fit(X)

        #labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        np.save('../data/processed/centroids_{}.npy'.format(K), np.array(centroids))

if __name__ == '__main__':
    main()