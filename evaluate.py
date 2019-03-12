import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import gen_data
from model import SimilarityModel
from utils import process_testing_samples, process_samples, ranking_sequence
from config import CONFIG as conf

model_path = conf['model_path']
batch_size = conf['batch_size']
device = conf['device']

def compute_diff_scores(model, samples, batch_size, all_relations, device):
    #testing_data = testing_data[0:100]
    for i in range((len(samples)-1)//batch_size+1):
        samples = samples[i*batch_size:(i+1)*batch_size]
        questions, relations, relation_set_lengths = \
            process_samples(samples, all_relations, device)
        model.init_hidden(device, sum(relation_set_lengths))
        ranked_questions, reverse_question_indexs = \
            ranking_sequence(questions)
        ranked_relations, reverse_relation_indexs =\
            ranking_sequence(relations)
        question_lengths = [len(question) for question in ranked_questions]
        relation_lengths = [len(relation) for relation in ranked_relations]
        #print(ranked_questions)
        pad_questions = torch.nn.utils.rnn.pad_sequence(ranked_questions)
        pad_relations = torch.nn.utils.rnn.pad_sequence(ranked_relations)
        all_scores = model(pad_questions, pad_relations, device,
                           reverse_question_indexs, reverse_relation_indexs,
                           question_lengths, relation_lengths)
        start_index = 0
        diff_scores = []
        #print('len of relation_set:', len(relation_set_lengths))
        for j in range(len(relation_set_lengths)):
            length = relation_set_lengths[j]
            '''
            cand_indexs = samples[j][1]
            gold_pos = np.where(np.array(cand_indexs)
                                == gold_relation_indexs[j])[0]
            print('gold pos', gold_pos)
            print('gold_index', gold_relation_indexs[j])
            print('cand index', cand_indexs)
            other_pos = np.where(np.array(cand_indexs)
                                 != gold_relation_indexs[j])[0]
            print('other_pos', other_pos)
            '''
            this_scores = all_scores[start_index:start_index + length]
            gold_score = this_scores[0]
            #print('gold score',gold_score)
            neg_scores = this_scores[1:]
            #print('neg score', neg_scores)
            diff_scores.append(gold_score - neg_scores.max())
            #print('scores:', all_scores[start_index:start_index+length])
            #print('cand indexs:', cand_indexs)
            #print('pred, true:',pred_index, gold_relation_indexs[j])
            start_index += length
        return diff_scores
# evaluate the model on the testing data
def evaluate_model(model, testing_data, batch_size, all_relations, device,
                   reverse_model=None):
    #print('start evaluate')
    num_correct = 0
    #testing_data = testing_data[0:100]
    for i in range((len(testing_data)-1)//batch_size+1):
        samples = testing_data[i*batch_size:(i+1)*batch_size]
        gold_relation_indexs, questions, relations, relation_set_lengths = \
            process_testing_samples(samples, all_relations, device)
        model.init_hidden(device, sum(relation_set_lengths))
        ranked_questions, reverse_question_indexs = \
            ranking_sequence(questions)
        ranked_relations, reverse_relation_indexs =\
            ranking_sequence(relations)
        question_lengths = [len(question) for question in ranked_questions]
        relation_lengths = [len(relation) for relation in ranked_relations]
        #print(ranked_questions)
        pad_questions = torch.nn.utils.rnn.pad_sequence(ranked_questions)
        pad_relations = torch.nn.utils.rnn.pad_sequence(ranked_relations)
        all_scores = model(pad_questions, pad_relations, device,
                           reverse_question_indexs, reverse_relation_indexs,
                           question_lengths, relation_lengths, reverse_model)
        start_index = 0
        pred_indexs = []
        #print('len of relation_set:', len(relation_set_lengths))
        for j in range(len(relation_set_lengths)):
            length = relation_set_lengths[j]
            cand_indexs = samples[j][1]
            pred_index = (cand_indexs[
                all_scores[start_index:start_index+length].argmax()])
            if pred_index == gold_relation_indexs[j]:
                num_correct += 1
            #print('scores:', all_scores[start_index:start_index+length])
            #print('cand indexs:', cand_indexs)
            #print('pred, true:',pred_index, gold_relation_indexs[j])
            start_index += length
    #print(cand_scores[-1])
    #print('num correct:', num_correct)
    #print('correct rate:', float(num_correct)/len(testing_data))
    return float(num_correct)/len(testing_data)

if __name__ == '__main__':
    model = torch.load(model_path)
    training_data, testing_data, valid_data,\
        all_relations, vocabulary,  embedding = gen_data()
    model.init_embedding(np.array(embedding))
    acc=evaluate_model(model, testing_data, batch_size, all_relations, device)
    print('accuracy:', acc)
