from matplotlib import pyplot as plt
import numpy as np
import csv
import pandas as pd
import sklearn
import torch as th
from sklearn.model_selection import train_test_split
import re
import random
from torch.nn.utils.rnn import pad_sequence


def clean_str(string, tolower=True):
    """
    Tokenization/string cleaning.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"((?<=\s)|^)[\d]+((?=\s)|$)", "<num>", string)
    if tolower:
        string = string.lower()
    return string.strip().split()


def join_data(stances_data, bodies_data, with_stance=True):
    headlines = []
    bodies = []
    stances = [] if with_stance else None

    for stance in stances_data.iterrows():
        stance = stance[1]
        headlines.append(stance['Headline'])
        body = bodies_data[bodies_data['Body ID'] == stance['Body ID']].iloc[0]
        bodies.append(body['articleBody'])
        
        if with_stance:
            stances.append(stance['Stance'])
        
    return headlines, stances, bodies


def clean_data(data):
    return list(map(clean_str, data))


stance_dict = {
    'agree': 0,
    'disagree': 1,
    'discuss': 2,
    'unrelated': 3
}

inv_stance_dict = dict((v, k) for (k, v) in stance_dict.items())



def transform_stances(stances):
    return [stance_dict[stance] for stance in stances]



def transform_back_stances(stances_num):
    return [inv_stance_dict[stance] for stance in stances_num]


def create_vocabulary(data, min_num_occur=10):
    vocab_dict = dict()
    vocab_freq_dict = dict()

    vocab_dict['<pad>'] = 0
    i = 1
    word_count = 0
    for sent in data:
        for word in sent:
            word_count += 1
            vocab_freq_dict[word] = vocab_freq_dict.get(word, 0) + 1
            if word not in vocab_dict and vocab_freq_dict[word] >= min_num_occur:
                vocab_dict[word] = i
                i += 1
                
    vocab_dict['<unk>'] = i
    
    return vocab_dict, vocab_freq_dict, word_count


def data_to_num_tensor(vocab_dict, data):
    num_tensor = [
        th.tensor(
            [
                (vocab_dict[word] if word in vocab_dict else vocab_dict['<unk>'])
                for word in sent
            ],
            dtype=th.long
        )
        for sent in data
    ]

    return pad_sequence(num_tensor, batch_first=True, padding_value=vocab_dict['<pad>'])


def preprocess_data(stances_data, bodies_data, with_stances=False):
    headlines, stances, bodies, body_ids = join_data(stances_data, bodies_data, with_stances)
    headlines_clean = clean_data(headlines)
    bodies_clean = clean_data(bodies)
    
    return headlines_clean, stances, bodies_clean, body_ids


def transform_data_to_tensor(vocab_dict, headlines, stances, bodies):
    if stances is not None:
        stances = th.tensor(transform_stances(stances), dtype=th.int64)
    
    headlines_tensor = data_to_num_tensor(vocab_dict, headlines)
    bodies_tensor = data_to_num_tensor(vocab_dict, bodies)
    
    return headlines_tensor, stances, bodies_tensor
    
    
def extract_data(stances_data, bodies_data, test_size=0.2, min_num_occur=10):
    headlines_clean, stances, bodies_clean, _ = preprocess_data(stances_data, bodies_data, with_stances=True)

    headlines_train, headlines_dev, stances_train, stances_dev, bodies_train, bodies_dev = train_test_split(
        headlines_clean,
        stances_clean,
        bodies_clean,
        test_size=test_size,
        stratify=stances_clean
    )
    
    vocab_dict, _, _ = create_vocabulary(headlines_train + bodies_train, min_num_occur=min_num_occur)
    
    return {
        'dict': vocab_dict,
        'train': transform_data_to_tensor(vocab_dict, headlines_train, stances_train, bodies_train),
        'dev': transform_data_to_tensor(vocab_dict, headlines_dev, stances_dev, bodies_dev)
    }


def transform_data(stances_data, bodies_data, vocab_dict, with_stances=False):
    headlines_clean, stances, bodies_clean = preprocess_data(stances_data, bodies_data, with_stances=with_stances)
    
    return transform_data_to_tensor(vocab_dict, headlines_clean, stances, bodies_clean)