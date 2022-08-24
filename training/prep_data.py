# -*- coding: utf-8 -*-
# 💾⚙️🔮

__author__ = "Daulet N., Robert Geislinger"
__email__ = "daulet.nurmanbetov@gmail.com, github@crpykng.de"

import re
import json
import random
import pandas as pd

VALID_LABELS = ['OU', 'OO', '.O', '!O', ',O', '.U', '!U', ',U', ':O', ';O', ':U', "'O", '-O', '?O', '?U']

def create_train_datasets():
    print('start processing')
    for i in 'abcdefghijk':
        i = 'subs_norm1_puncta' + i
        print(f'actual inputfile: {i}')
        all_records = create_rpunct_dataset(i)
        records = create_training_samples(all_records)
        eval_set = records[-int(len(records) * 0.10):]
        train_set = records[:int(len(records) * 0.90)]
        create_text_file(train_set, 'subs_norm1_trainset.txt', append=True)
        create_text_file(eval_set, 'subs_norm1_evalset.txt', append=True)


def create_record(row, valid_labels):
    """
    Create labels for Punctuation Restoration task for each token.
    """
    pattern = re.compile("[\W_]+")
    new_obs = []
    observation = row.replace('\\n', ' ').split()

    for obs in observation:
        text_obs = obs.lower()
        text_obs = pattern.sub('', text_obs)

        if not text_obs:
            continue
        if not obs[-1].isalnum():
            new_lab = obs[-1]
        else:
            new_lab = "O"
        if obs[0].isupper():
            new_lab += "U"
        else:
            new_lab += "O"
        if new_lab in valid_labels:
            new_obs.append({'sentence_id': 0, 'words': text_obs, 'labels': new_lab})
    return new_obs


def create_rpunct_dataset(orig_data):
    print('create dataset')
    df = pd.DataFrame()
    input = ''
    with open(orig_data) as f:
        input=f.read()
    inputA = input.split('\n')
    df = pd.DataFrame(inputA)
    all_records = []
    for i in df[0]:
        records = create_record(i, valid_labels=VALID_LABELS)
        all_records.extend(records)
    return all_records


def create_training_samples(all_records):
    """
    Given a looong list of tokens, splits them into 500 token chunks
    thus creating observations. This is for fine-tuning with simpletransformers
    later on.
    """
    print('create training samples')
    random.seed(1337)
    observations = []

    obs = create_tokenized_obs(all_records)
    full_data = pd.DataFrame(all_records)

    for i in obs:
        data_slice = full_data.iloc[i[0]:i[1], ]
        observations.append(data_slice.values.tolist())

    random.shuffle(observations)
    return observations


def create_tokenized_obs(input_list, num_toks=500, offset=250):
    """
    Given a large set of tokens, determines splits of
    500 token sized observations, with an offset(sliding window) of 250 tokens.
    It is important that first token is capitalized and we fed as many tokens as possible.
    In a real use-case we will not know where splits are so we'll just feed all tokens till limit.
    """
    print('create tokenized observations')
    start = -1
    loop_end = -1
    appends = []
    for ix, i in enumerate(input_list):
        if ix == loop_end:
            start = -1
        if i['labels'][-1] == "U" and start == -1:
            start = ix
            end = ix + num_toks
            appends.append((start, end))
            loop_end = start + offset
            
    return appends


def load_datasets(dataset_paths):
    """
    Given a list of data paths returns a single data object containing all data slices
    """
    print('load datasets')
    token_data = []
    for d_set in dataset_paths:
        with open(d_set, 'r') as fp:
            data_slice = json.load(fp)
        token_data.extend(data_slice)
        del data_slice
    return token_data

def get_label_stats(dataset):
    """
    Generates frequency of different labels in the dataset.
    """
    print('get label stats')
    calcs = {}
    for i in dataset:
        for tok in i:
            if tok[2] not in calcs.keys():
                calcs[tok[2]] = 1
            else:
                calcs[tok[2]] += 1
    print(calcs)
    return calcs

def create_text_file(dataset, name, append=False):
    """
    Create Connl ner format file
    """
    if append:
        mode = 'a'
    else:
        append = 'w'    
    print('create text file')
    with open(name, mode) as fp:
        for obs in dataset:
            for tok in obs:
                word = tok[1].lower().replace('!', '').replace('.', '').replace(',', '').replace(':', '').replace(';', '').replace("'", '').replace('-', '').replace('?', '')
                line = word + " " + tok[2] + '\n'
                fp.write(line)
            fp.write('\n')


if __name__ == "__main__":
    output_file_names = create_train_datasets()
