import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import random
import time
from sklearn.cluster import KMeans
from sklearn import preprocessing  # to normalise existing X

from data import gen_data
from model import SimilarityModel
from utils import process_testing_samples, process_samples, ranking_sequence,\
    get_grad_params, copy_param_data, copy_grad_data
from evaluate import evaluate_model, compute_diff_scores
from data_partition import cluster_data
from config import CONFIG as conf
from train import train
from compute_rel_embed import compute_rel_embed
from alignment_model import update_alignment_model
from alignment_model import AlignmentModel

embedding_dim = conf['embedding_dim']
hidden_dim = conf['hidden_dim']
batch_size = conf['batch_size']
device = conf['device']
num_clusters = conf['num_clusters']
lr = conf['learning_rate']
model_path = conf['model_path']
epoch = conf['epoch']
random_seed = conf['random_seed']
task_memory_size = conf['task_memory_size']
loss_margin = conf['loss_margin']
sequence_times = conf['sequence_times']
num_cands = conf['num_cands']
num_steps = conf['num_steps']
num_contrain = conf['num_constrain']
data_per_constrain = conf['data_per_constrain']

def split_data(data_set, cluster_labels, num_clusters, shuffle_index):
    splited_data = [[] for i in range(num_clusters)]
    for data in data_set:
        cluster_number = cluster_labels[data[0]]
        index_number = shuffle_index[cluster_number]
        splited_data[index_number].append(data)
    return splited_data

# remove unseen relations from the dataset
def remove_unseen_relation(dataset, seen_relations):
    cleaned_data = []
    for data in dataset:
        neg_cands = [cand for cand in data[1] if cand in seen_relations]
        if len(neg_cands) > 0:
            cleaned_data.append([data[0], neg_cands, data[2]])
        else:
            #Cause FewRel has much less relations,it is easy that no negative
            #candidate is found. So we keep last two relations
            #in the negative candidate pool when it happens
            if conf['is_FewRel']:
                cleaned_data.append([data[0], data[1][-2:], data[2]])
            pass
    return cleaned_data

def print_list(result):
    for num in result:
        sys.stdout.write('%.3f, ' %num)
    print('')

def get_que_embed(model, sample_list, all_relations, alignment_model,
                  before_alignment=False):
    ret_que_embeds = []
    for i in range((len(sample_list)-1)//batch_size+1):
        samples = sample_list[i*batch_size:(i+1)*batch_size]
        questions = []
        for item in samples:
            this_question = torch.tensor(item[2], dtype=torch.long).to(device)
            questions.append(this_question)
        #print(len(questions))
        model.init_hidden(device, len(questions))
        ranked_questions, alignment_question_indexs = \
            ranking_sequence(questions)
        question_lengths = [len(question) for question in ranked_questions]
        #print(ranked_questions)
        pad_questions = torch.nn.utils.rnn.pad_sequence(ranked_questions)
        que_embeds = model.compute_que_embed(pad_questions, question_lengths,
                                             alignment_question_indexs,
                                             alignment_model, before_alignment)
        ret_que_embeds.append(que_embeds.detach().cpu().numpy())
    return np.concatenate(ret_que_embeds)

# get the embedding of relations. If before_alignment is False, then the
# embedding after the alignment model will be returned. Otherwise, the embedding
# before the alignment model will be returned
def get_rel_embed(model, sample_list, all_relations, alignment_model,
                  before_alignment=False):
    ret_rel_embeds = []
    for i in range((len(sample_list)-1)//batch_size+1):
        samples = sample_list[i*batch_size:(i+1)*batch_size]
        relations = []
        for item in samples:
            this_relation = torch.tensor(all_relations[item[0]],
                                         dtype=torch.long).to(device)
            relations.append(this_relation)
        #print(len(relations))
        model.init_hidden(device, len(relations))
        ranked_relations, alignment_relation_indexs = \
            ranking_sequence(relations)
        relation_lengths = [len(relation) for relation in ranked_relations]
        #print(ranked_relations)
        pad_relations = torch.nn.utils.rnn.pad_sequence(ranked_relations)
        rel_embeds = model.compute_rel_embed(pad_relations, relation_lengths,
                                             alignment_relation_indexs,
                                             alignment_model, before_alignment)
        ret_rel_embeds.append(rel_embeds.detach().cpu().numpy())
    return np.concatenate(ret_rel_embeds)

def update_rel_cands(memory_data, all_seen_cands, rel_embeds):
    if len(memory_data) >0:
        for this_memory in memory_data:
            for sample in this_memory:
                valid_rels = [rel for rel in all_seen_cands if rel!=sample[0]]
                sample[1] = random.sample(valid_rels,
                                        min(num_cands,len(valid_rels)))

# Use K-Means to select what samples to save
def select_data(model, samples, num_sel_data, all_relations, alignment_model):
    que_embeds = get_que_embed(model, samples, all_relations, alignment_model)
    que_embeds = preprocessing.normalize(que_embeds)
    #print(que_embeds[:5])
    num_clusters = min(num_sel_data, len(samples))
    distances = KMeans(n_clusters=num_clusters,
                    random_state=0).fit_transform(que_embeds)
    selected_samples = []
    for i in range(num_clusters):
        sel_index = np.argmin(distances[:,i])
        selected_samples.append(samples[sel_index])
    return selected_samples

def get_dis2mean(embeds, sel_index, cand_index, mean_embed):
    this_index = sel_index + [cand_index]
    this_embed = np.mean(embeds[this_index], 0)
    return np.linalg.norm(this_embed-mean_embed)

def select_data_icarl(model, samples, num_sel_data,
                      all_relations, alignment_model):
    que_embeds = get_que_embed(model, samples, all_relations, alignment_model)
    que_embeds = preprocessing.normalize(que_embeds)
    #print(que_embeds[:5])
    mean_embed = que_embeds.mean(0)
    sel_index = []
    sample_len = len(samples)
    for i in range(min(num_sel_data, sample_len)):
        dis_list = [get_dis2mean(que_embeds, sel_index, cand, mean_embed)
                    for cand in range(sample_len)]
        sel_index.append(np.argmin(dis_list))
    return [samples[i] for i in sel_index]

def compute_cos_similarity(a, b):
    a_t = torch.from_numpy(a)
    b_t = torch.from_numpy(b)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    result = cos(a_t, b_t)
    return np.mean(result.numpy())

def run_sequence(training_data, testing_data, valid_data, all_relations,
                 vocabulary,embedding, cluster_labels, num_clusters,
                 shuffle_index, rel_embeds):
    splited_training_data = split_data(training_data, cluster_labels,
                                       num_clusters, shuffle_index)
    splited_valid_data = split_data(valid_data, cluster_labels,
                                    num_clusters, shuffle_index)
    splited_test_data = split_data(testing_data, cluster_labels,
                                   num_clusters, shuffle_index)
    seen_relations = []
    current_model = None
    #'alignment_model' is correspondin to the alignment linear model
    alignment_model = None
    memory_data = []
    memory_que_embed = []
    memory_rel_embed = []
    sequence_results = []
    result_whole_test = []
    all_seen_rels = []
    for i in range(num_clusters):
        for data in splited_training_data[i]:
            if data[0] not in seen_relations:
                seen_relations.append(data[0])
        current_train_data = remove_unseen_relation(splited_training_data[i],
                                                    seen_relations)
        current_valid_data = remove_unseen_relation(splited_valid_data[i],
                                                    seen_relations)
        current_test_data = []
        for j in range(i+1):
            current_test_data.append(
                remove_unseen_relation(splited_test_data[j], seen_relations))
        one_memory_data = []
        for this_sample in current_train_data:
            if this_sample[0] not in all_seen_rels:
                all_seen_rels.append(this_sample[0])
        update_rel_cands(memory_data, all_seen_rels, rel_embeds)
        to_train_data = current_train_data
        current_model= train(to_train_data, current_valid_data,
                             vocabulary, embedding_dim, hidden_dim,
                             device, batch_size, lr, embedding,
                             all_relations, current_model, epoch,
                             memory_data, loss_margin, alignment_model)
        #memory_data.append(current_train_data[-task_memory_size:])
        memory_data.append(select_data(current_model, current_train_data,
                                       task_memory_size, all_relations,
                                       alignment_model))
        #memory_data.append(select_data_icarl(current_model, current_train_data,
        #                               task_memory_size, all_relations,
        #                               alignment_model))
        memory_que_embed.append(get_que_embed(current_model, memory_data[-1],
                                              all_relations, alignment_model))
        memory_rel_embed.append(get_rel_embed(current_model, memory_data[-1],
                                              all_relations, alignment_model))
        to_train_data = []
        for this_memory in memory_data:
            to_train_data += this_memory
        if len(memory_data) > 1:
            cur_que_embed = [get_que_embed(current_model, this_memory,
                                           all_relations, alignment_model, True)
                             for this_memory in
                             memory_data]
            cur_rel_embed = [get_rel_embed(current_model, this_memory,
                                           all_relations, alignment_model, True)
                             for this_memory in
                             memory_data]
            alignment_model = update_alignment_model(alignment_model, cur_que_embed,
                                                 cur_rel_embed,
                                                 memory_que_embed,
                                                 memory_rel_embed)
            memory_que_embed = [get_que_embed(current_model, this_memory,
                                           all_relations, alignment_model, False)
                             for this_memory in
                             memory_data]
            memory_rel_embed = [get_rel_embed(current_model, this_memory,
                                           all_relations, alignment_model, False)
                             for this_memory in
                             memory_data]
        results = [evaluate_model(current_model, test_data, batch_size,
                                  all_relations, device, alignment_model)
                   for test_data in current_test_data]
        print_list(results)
        sequence_results.append(np.array(results))
        result_whole_test.append(evaluate_model(current_model,
                                                testing_data, batch_size,
                                                all_relations, device,
                                                alignment_model))
    print('test set size:', [len(test_set) for test_set in current_test_data])
    return sequence_results, result_whole_test

def print_avg_results(all_results):
    avg_result = []
    for i in range(len(all_results[0])):
        avg_result.append(np.average([result[i] for result in all_results], 0))
    for line_result in avg_result:
        print_list(line_result)
    return avg_result

def print_avg_cand(sample_list):
    cand_lengths = []
    for sample in sample_list:
        cand_lengths.append(len(sample[1]))
    print('avg cand size:', np.average(cand_lengths))

if __name__ == '__main__':
    random_seed = int(sys.argv[1])
    training_data, testing_data, valid_data, all_relations, vocabulary, \
        embedding=gen_data()
    cluster_labels, rel_features = cluster_data(num_clusters)
    to_use_embed = rel_features
    #to_use_embed = bert_rel_features
    random.seed(random_seed)
    start_time = time.time()
    all_results = []
    result_all_test_data = []
    for i in range(sequence_times):
        shuffle_index = list(range(num_clusters))
        random_seed = int(sys.argv[1]) + 100*i
        random.seed(random_seed)
        #random.seed(random_seed+100*i)
        random.shuffle(shuffle_index)
        sequence_results, result_whole_test = run_sequence(
            training_data, testing_data, valid_data, all_relations,
            vocabulary, embedding, cluster_labels, num_clusters, shuffle_index,
            to_use_embed)
        all_results.append(sequence_results)
        result_all_test_data.append(result_whole_test)
    avg_result_all_test = np.average(result_all_test_data, 0)
    for result_whole_test in result_all_test_data:
        print_list(result_whole_test)
    print_list(avg_result_all_test)
    print_avg_results(all_results)
    end_time = time.time()
    elapsed_time = (end_time - start_time) / sequence_times
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
