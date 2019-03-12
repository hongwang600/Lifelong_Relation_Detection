import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import quadprog
import random
import time

from data import gen_data, read_origin_relation
from model import SimilarityModel
from utils import process_testing_samples, process_samples, ranking_sequence,\
    copy_grad_data, get_grad_params
from evaluate import evaluate_model
from config import CONFIG as conf

embedding_dim = conf['embedding_dim']
hidden_dim = conf['hidden_dim']
batch_size = conf['batch_size']
model_path = conf['model_path']
device = conf['device']
lr = conf['learning_rate']
loss_margin = conf['loss_margin']
random.seed(100)
origin_relation_names = read_origin_relation()


def feed_samples(model, samples, loss_function, all_relations, device,
                 alignment_model=None):
    questions, relations, relation_set_lengths = process_samples(
        samples, all_relations, device)
    ranked_questions, alignment_question_indexs = \
        ranking_sequence(questions)
    ranked_relations, alignment_relation_indexs =\
        ranking_sequence(relations)
    question_lengths = [len(question) for question in ranked_questions]
    relation_lengths = [len(relation) for relation in ranked_relations]
    pad_questions = torch.nn.utils.rnn.pad_sequence(ranked_questions)
    pad_relations = torch.nn.utils.rnn.pad_sequence(ranked_relations)
    pad_questions = pad_questions.to(device)
    pad_relations = pad_relations.to(device)

    model.zero_grad()
    if alignment_model is not None:
        alignment_model.zero_grad()
    model.init_hidden(device, sum(relation_set_lengths))
    all_scores = model(pad_questions, pad_relations, device,
                       alignment_question_indexs, alignment_relation_indexs,
                       question_lengths, relation_lengths, alignment_model)
    all_scores = all_scores.to('cpu')
    pos_scores = []
    neg_scores = []
    pos_index = []
    start_index = 0
    for length in relation_set_lengths:
        pos_index.append(start_index)
        pos_scores.append(all_scores[start_index].expand(length-1))
        neg_scores.append(all_scores[start_index+1:start_index+length])
        start_index += length
    pos_scores = torch.cat(pos_scores)
    neg_scores = torch.cat(neg_scores)
    alignment_model_criterion = nn.MSELoss()

    loss = loss_function(pos_scores, neg_scores,
                         torch.ones(sum(relation_set_lengths)-
                                    len(relation_set_lengths)))
    loss.backward()
    return all_scores, loss

def train(training_data, valid_data, vocabulary, embedding_dim, hidden_dim,
          device, batch_size, lr, embedding, all_relations,
          model=None, epoch=100, memory_data=[], loss_margin=0.5,
          alignment_model=None):
    if model is None:
        torch.manual_seed(100)
        model = SimilarityModel(embedding_dim, hidden_dim, len(vocabulary),
                                np.array(embedding), 1, device)
    loss_function = nn.MarginRankingLoss(loss_margin)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0
    memory_index = 0
    for epoch_i in range(epoch):
        for i in range((len(training_data)-1)//batch_size+1):
            samples = training_data[i*batch_size:(i+1)*batch_size]
            seed_rels = []
            for item in samples:
                if item[0] not in seed_rels:
                    seed_rels.append(item[0])

            if len(memory_data) > 0:
                all_seen_data = []
                for this_memory in memory_data:
                    all_seen_data+=this_memory
                memory_batch = memory_data[memory_index]
                scores, loss = feed_samples(model, memory_batch,
                                            loss_function,
                                            all_relations, device,
                                            alignment_model)
                optimizer.step()
                memory_index = (memory_index+1)%len(memory_data)
            scores, loss = feed_samples(model, samples, loss_function,
                                        all_relations, device, alignment_model)
            optimizer.step()
            del scores
            del loss
    return model

if __name__ == '__main__':
    training_data, testing_data, valid_data, all_relations, vocabulary, \
        embedding=gen_data()
    train(training_data, valid_data, vocabulary, embedding_dim, hidden_dim,
          device, batch_size, lr, model_path, embedding, all_relations,
          model=None, epoch=100)
