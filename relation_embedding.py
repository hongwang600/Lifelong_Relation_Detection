import numpy as np
import wordninja
from sklearn.decomposition import PCA
#from matplotlib import pyplot
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from config import CONFIG as conf

sq_relation_name_file = conf['relation_file']
sq_train_file = conf['training_file']
sq_valid_file = conf['valid_file']
sq_test_file = conf['test_file']
glove_input_file = conf['glove_file']

# read the indexs of relations in a given file
def read_relations_index(file_name):
    relation_pool = []
    with open(file_name) as in_file:
        for line in in_file:
            relation_number = int(line.split("\t")[0])
            if relation_number not in relation_pool:
                relation_pool.append(relation_number)
    return relation_pool

# extract the names of relations given their indexs, note that the index starts
# from 1
def read_relation_names(file_name, relation_index):
    all_relations = []
    with open(file_name) as in_file:
        for line in in_file:
            # remove "\n" for each line
            all_relations.append(line.split("\n")[0])
    relation_names = [all_relations[num-1] for num in relation_index]
    return relation_names

# read the embeddings for a given vocabulary
def read_glove_embeddings(glove_input_file):
    glove_dict = {}
    with open(glove_input_file) as in_file:
        for line in in_file:
            values = line.split()
            word = values[0]
            glove_dict[word] = np.asarray(values[1:], dtype='float32')
    return glove_dict

def split_relation_into_words(relation):
    word_list = []
    # some relation will have fours parts, where the first part looks like
    # "base". We only choose the last three parts
    for word_seq in relation.split("/")[-3:]:
        for word in word_seq.split("_"):
            word_list += wordninja.split(word)
    return word_list

# generate the vocabulary of these realtions
def gen_vocabulary(relation_names):
    vocabulary = []
    for relation in relation_names:
        word_list = split_relation_into_words(relation)
        vocabulary += [word.lower()
                       for word in word_list if word not in vocabulary]
    return vocabulary


# get the embedding for a relation
def get_embedding(relation_name, glove_embeddings):
    word_list = split_relation_into_words(relation_name)
    relation_embeddings = []
    for word in word_list:
        if word.lower() in glove_embeddings:
            relation_embeddings.append(glove_embeddings[word.lower()])
        else:
            print(word,"is not contained in glove vocabulary")
    return np.mean(relation_embeddings, 0)

def gen_relation_embedding():
    train_relation_index = read_relations_index(sq_train_file)
    #print(train_relation_index)
    valid_relation_index = read_relations_index(sq_valid_file)
    test_relation_index = read_relations_index(sq_test_file)
    # Here list(a) will copy items in a. list.copy() not availabel in python2
    relation_index = list(train_relation_index)
    for index in test_relation_index+valid_relation_index:
        if index not in relation_index:
            relation_index.append(index)
    relation_index = np.array(relation_index)
    #print(relation_index[-1])
    relation_names = read_relation_names(sq_relation_name_file, relation_index)
    #vocabulary = gen_vocabulary(relation_names)
    glove_embeddings = read_glove_embeddings(glove_input_file)
    #print(glove_embeddings)
    #print(glove_embeddings['dancer'])
    #print(vocabulary)
    #print(relation_names[-1])
    relation_embeddings = []
    for relation in relation_names:
        relation_embeddings.append(get_embedding(relation,
                                                 glove_embeddings))
    '''
    with open('relations.txt', 'w') as file_out:
        for item in relation_names:
            file_out.write('%s\n' %item)
    '''
    relation_embeddings = np.asarray(relation_embeddings)
    '''
    relation_dict = {}
    for i in range(len(relation_index)):
        relation_dict[relation_index[i]] = i
    '''
    return relation_names, relation_index, relation_embeddings
    #print(len(relation_embeddings[0]))
    #np.save('relation_embeddings.npy', relation_embeddings)
    #relation_embeddings = np.load('relation_embeddings.npy')
    #print(len(relation_embeddings[0]))
