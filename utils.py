
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import CONFIG as conf

# process the data by adding questions
def process_testing_samples(sample_list, all_relations, device):
    questions = []
    relations = []
    gold_relation_indexs = []
    relation_set_lengths = []
    for sample in sample_list:
        question = torch.tensor(sample[2], dtype=torch.long).to(device)
        #print(relations[sample[0]])
        #print(sample)
        gold_relation_indexs.append(sample[0])
        neg_relations = [torch.tensor(all_relations[index],
                                      dtype=torch.long).to(device)
                         for index in sample[1]]
        relation_set_lengths.append(len(neg_relations))
        relations += neg_relations
        #questions += [question for i in range(relation_set_lengths[-1])]
        questions += [question] * relation_set_lengths[-1]
    return gold_relation_indexs, questions, relations, relation_set_lengths

# process the data by adding questions
def process_samples(sample_list, all_relations, device):
    questions = []
    relations = []
    relation_set_lengths = []
    for sample in sample_list:
        question = torch.tensor(sample[2], dtype=torch.long).to(device)
        #print(relations[sample[0]])
        #print(sample)
        pos_relation = torch.tensor(all_relations[sample[0]],
                                    dtype=torch.long).to(device)
        neg_relations = [torch.tensor(all_relations[index],
                                      dtype=torch.long).to(device)
                         for index in sample[1]]
        relation_set_lengths.append(len(neg_relations)+1)
        relations += [pos_relation]+ neg_relations
        #questions += [question for i in range(relation_set_lengths[-1])]
        questions += [question] * relation_set_lengths[-1]
    return questions, relations, relation_set_lengths

def ranking_sequence(sequence):
    word_lengths = torch.tensor([len(sentence) for sentence in sequence])
    rankedi_word, indexs = word_lengths.sort(descending = True)
    ranked_indexs, inverse_indexs = indexs.sort()
    #print(indexs)
    sequence = [sequence[i] for i in indexs]
    return sequence, inverse_indexs

def get_grad_params(model):
    grad_params = []
    for param in model.parameters():
        if param.requires_grad:
            grad_params.append(param)
    return grad_params

def copy_param_data(params):
    copy_params = []
    for param in params:
        copy_params.append(param.data.clone())
    return copy_params


def copy_grad_data(model):
    params = get_grad_params(model)
    param_grads = []
    for param in params:
        param_grads.append(param.grad.view(-1).clone())
    return torch.cat(param_grads)
