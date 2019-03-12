import numpy as np
import wordninja
import re
from config import CONFIG as conf

relation_file = conf['relation_file']
training_file = conf['training_file']
test_file = conf['test_file']
valid_file = conf['valid_file']
#glove_file = "./data/glove.6B.300d.txt"
glove_file = conf['glove_file']
embedding_size = conf['embedding_dim']
random_seed = conf['random_seed']

# remove return symbol
def remove_return_sym(str):
    return str.split('\n')[0]

# get the relation names from the file
def read_relations(relation_file):
    relation_list = []
    relation_list.append('/fill/fill/fill')
    with open(relation_file) as file_in:
        for line in file_in:
            relation_list.append(remove_return_sym(line))
    return relation_list

# extract the glove vocabulary and glove embedding from the file
def read_glove(glove_file):
    glove_vocabulary = []
    glove_embedding = {}
    with open(glove_file) as file_in:
        for line in file_in:
            items = line.split()
            word = items[0]
            glove_vocabulary.append(word)
            glove_embedding[word] = np.asarray(items[1:], dtype='float32')
    return glove_vocabulary, glove_embedding

# read the training/testing/valid sample from file
def read_samples(sample_file):
    sample_data = []
    with open(sample_file) as file_in:
        for line in file_in:
            items = line.split('\t')
            if(len(items[0])>0):
                relation_ix = int(items[0])
                #print(items[1].split())
                if items[1] != 'noNegativeAnswer':
                    candidate_ixs = [int(ix) for ix in items[1].split()]
                    question = remove_return_sym(items[2]).split()
                    sample_data.append([relation_ix, candidate_ixs, question])
    return sample_data

# concat words by using '_'
def concat_words(words):
    if len(words) > 0:
        return_str = words[0]
        for word in words[1:]:
            return_str += '_' + word
        return return_str
    else:
        return ''

# split the relation into words together with relation name, eg. birth_place_of
# will be turned into [birth, place, of, birth_place_of]
def split_relation_into_words(relation, glove_vocabulary):
    word_list = []
    relation_list = []
    # some relation will have fours parts, where the first part looks like
    # "base". We only choose the last three parts
    for word_seq in relation.split("/")[-3:]:
        new_word_list = []
        #for word in word_seq.split("_"):
        for word in re.findall(r"[\w']+", word_seq):
            #print(word)
            if word not in glove_vocabulary:
                new_word_list += wordninja.split(word)
            else:
                new_word_list += [word]
        #print(new_word_list)
        word_list += new_word_list
        relation_list.append(concat_words(new_word_list))
    return word_list+relation_list

# some words are put together, such computerscience. Need to split these words
# in the samples, and will split the relation
def clean_relations(relation_list, glove_vocabulary):
    cleaned_relations = []
    for relation in relation_list:
        cleaned_relations.append(split_relation_into_words(relation, glove_vocabulary))
    return cleaned_relations

# build the vocabulary and embedding from the relations and questions based on
# the glove embeddings
def build_vocabulary_embedding(relation_list, all_samples, glove_embedding,
                               embedding_size):
    vocabulary = {}
    embedding = []
    index = 0
    np.random.seed(100)
    for relation in relation_list:
        for word in relation:
            if word not in vocabulary:
                vocabulary[word] = index
                index += 1
                # init the word that are not in glove vocabulary randomly
                if word in glove_embedding:
                    embedding.append(glove_embedding[word])
                else:
                    embedding.append(np.random.rand(embedding_size))
    for sample in all_samples:
        question = sample[2]
        for word in question:
            if word not in vocabulary:
                vocabulary[word] = index
                index += 1
                # init the word that are not in glove vocabulary randomly
                if word in glove_embedding:
                    embedding.append(glove_embedding[word])
                else:
                    #print(word)
                    embedding.append(np.random.rand(embedding_size))

    return vocabulary, embedding

# transform the word in the list into the index in the vocabulary
def words2indexs(word_list, vocabulary):
    index_list = []
    for word in word_list:
        index_list.append(vocabulary[word])
    return index_list

# transform the words in the relations into index in the vocabulary
def transform_relations(relation_list, vocabulary):
    relation_ixs = []
    for relation in relation_list:
        relation_ixs.append(words2indexs(relation, vocabulary))
    return relation_ixs

# transform the words in the questions into index of the vocabulary
def transform_questions(sample_list, vocabulary):
    for sample in sample_list:
        sample[2] = words2indexs(sample[2], vocabulary)
    return sample_list

def read_origin_relation():
    relation_list = read_relations(relation_file)
    return relation_list

# generate the training, valid, test data
#def gen_data(relation_file, training_file, test_file, valid_file, glove_file):
def gen_data():
    relation_list = read_relations(relation_file)
    #print(relation_list[1:10])
    glove_vocabulary, glove_embedding = read_glove(glove_file)
    #print(glove_vocabulary[0:10])
    #print(glove_embedding['of'])
    training_data = read_samples(training_file)
    #print(training_data[0])
    testing_data = read_samples(test_file)
    valid_data = read_samples(valid_file)
    all_samples = training_data + testing_data + valid_data
    #print(training_data[0])
    cleaned_relations = clean_relations(relation_list, glove_vocabulary)
    #print(cleaned_relations)
    vocabulary, embedding = build_vocabulary_embedding(cleaned_relations,
                                                       all_samples,
                                                       glove_embedding,
                                                       embedding_size)
    #print(embedding)
    #print(vocabulary)
    #print(len(vocabulary), len(embedding))
    relation_numbers = transform_relations(cleaned_relations, vocabulary)
    #print(relation_numbers[0:10])
    training_data = transform_questions(training_data, vocabulary)
    #print(training_data[0:10])
    testing_data = transform_questions(testing_data, vocabulary)
    valid_data = transform_questions(valid_data, vocabulary)
    return training_data, testing_data, valid_data, relation_numbers,\
        vocabulary,embedding

if __name__ == '__main__':
    gen_data(relation_file, training_file, test_file, valid_file, glove_file)
