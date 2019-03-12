import numpy as np
from sklearn.cluster import KMeans
from relation_embedding import gen_relation_embedding
from config import CONFIG as conf

np.set_printoptions(threshold=np.inf)
#num_clusters = 50
train_data_path = conf['training_file']
relation_path = conf['relation_file']


def cluster_data(num_clusters=20):
    relation_names, relation_index, relation_embeddings = \
        gen_relation_embedding()
    kmeans = KMeans(n_clusters=num_clusters,
                    random_state=0).fit(relation_embeddings)
    #print(kmeans.inertia_)
    labels = kmeans.labels_
    rel_embed = {}
    cluster_index = {}
    for i in range(len(relation_index)):
        cluster_index[relation_index[i]] = labels[i]
        rel_embed[relation_index[i]] = relation_embeddings[i]
    rel_index = np.asarray(list(relation_index))
    return cluster_index, rel_embed

if __name__ == '__main__':
    cluster_index = cluster_data(num_clusters=20)
    print(cluster_index)
