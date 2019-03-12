import numpy as np
from sklearn.decomposition import PCA
#from matplotlib import pyplot
from data import gen_data
from config import CONFIG as conf

bert_feature_file = conf['bert_feature_file']

# visualize using PCA
def visualize_PCA(X, names):
    start_x = -2
    start_y = 1
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1], 1)
    for i, text in enumerate(names):
        #if abs(result[i,0]-start_x) < 0.2 and abs(result[i,1]-start_y) < 0.2:
        pyplot.annotate(text, (result[i,0], result[i,1]))
    pyplot.show()
    '''
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
        pyplot.show()
    '''

def read_relation_name(file_name):
    ret_names = []
    with open(file_name) as file_in:
        for line in file_in:
            ret_names.append(line[:-2])
    return ret_names

def read_embedding(file_name):
    ret_np = []
    with open(file_name) as file_in:
        for line in file_in:
            ret_np.append(np.fromstring(line[1:-2], dtype=float, sep=','))
    #print(ret_np)
    return np.asarray(ret_np)

def compute_rel_embed(training_data, relation_names=None):
    que_rel_embeddings = read_embedding(bert_feature_file)
    rel_indexs = {}
    for i, sample in enumerate(training_data):
        rel = sample[0]
        if rel not in rel_indexs:
            rel_indexs[rel] = [i]
        else:
            rel_indexs[rel].append(i)
    '''
    for rel in rel_indexs:
        print(all_relations[rel])
        for index in rel_indexs[rel]:
            print(relation_names[index])
        break
        '''
    rel_embed = {}
    for rel in rel_indexs:
        que_rel_embeds = [que_rel_embeddings[i] for i in rel_indexs[rel]]
        #rel_embed[rel] = np.max(que_rel_embeds, 0)
        rel_embed[rel] = np.mean(que_rel_embeds, 0)
    rel_ids = rel_embed.keys()
    if relation_names is not None:
        rel_names = [relation_names[rel_indexs[i][0]].split('|||')[1]
                     for i in rel_ids]
        rel_embed_value = np.array(list(rel_embed.values()))
        return rel_names, rel_embed_value, rel_embed
    else:
        return rel_embed

if __name__ == "__main__":
    relation_names = read_relation_name('question_relation.txt')
    training_data, testing_data, valid_data, all_relations, vocabulary, \
        embedding=gen_data()
    rel_names, rel_embed_values,rel_embed=compute_rel_embed(training_data,
                                                            relation_names)
    visualize_PCA(rel_embed_values, rel_names)
    #num_samples_2_visual = len(relation_embeddings)
    #visualize_PCA(relation_embeddings[:num_samples_2_visual],
    #              relation_names[:num_samples_2_visual])
