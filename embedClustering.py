import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from sklearn.manifold import TSNE
import plotly.express as px
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from tqdm import tqdm

def average_embeddings(embeddings):
    averaged_embeddings = [np.mean(embedding, axis=0) for embedding in embeddings]
    return averaged_embeddings 

def tsneReduceEMB(averaged_embeddings):
    tsne = TSNE(n_components=2, **kwargs)
    reduced_embeddings = tsne.fit_transform(averaged_embeddings)
    return reduced_embeddings
    
def findEPS(reduced_embeddings, k=4):
    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(reduced_embeddings)
    distances, indices = nbrs.kneighbors(reduced_embeddings)

    distances = np.sort(distances[:, k-1], axis=0)
    plt.plot(distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel('k-distance')
    plt.title('k-distance graph to determine eps')
    plt.show()

    kneedle = KneeLocator(range(len(distances)), distances, S=1.0, curve='convex', direction='increasing')
    optimal_eps = distances[kneedle.elbow]
    return optimal_eps

def dbscanEMB(reduced_embeddings, eps=1.8, min_samples=4):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(reduced_embeddings)
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    return labels

def hdbscanEMB(reduced_embeddings, min_samples=4):
    db = HDBSCAN(min_samples=min_samples).fit(reduced_embeddings)
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    return labels

def plot_clusters(reduced_embeddings, text, labels):
    tsne_df = pd.DataFrame(reduced_embeddings, columns=['Component 1', 'Component 2'])
    tsne_df['text'] = text
    tsne_df['cluster'] = labels


    fig = px.scatter(
        tsne_df,
        x='Component 1',
        y='Component 2',
        color='cluster',
        hover_data=['text'],
        #color_continuous_scale=px.colors.diverging.BrBG
    )
    fig.show()
    return fig

def save_fig(fig):
    fig.write_html("interactive_plot.html")


