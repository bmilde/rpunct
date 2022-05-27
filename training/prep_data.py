# -*- coding: utf-8 -*-
# 💾⚙️🔮

__author__ = "Daulet N., Robert Geislinger"
__email__ = "daulet.nurmanbetov@gmail.com, github@crpykng.de"

import os
import re
import json
import random
import pandas as pd

def create_train_datasets():
    output_file_names = []
    # download_df()
    print('start processing')
    for i in 'abcdefghijk':
        i = 'subs_norm1_puncta' + i
        print(f'{i=}')
        name = i.split(".")[0]
        split_nm = name.split("_")[-1]
        df_name = name.split("_")[0]
        all_records = create_rpunct_dataset(i)
        output_file_names.append(f"{df_name}_{split_nm}.txt")
        create_training_samples(all_records, f"{df_name}_{split_nm}")
    return output_file_names


def download_df(dir_path=''):
    import tensorflow_datasets as tfds
    data_type = ['train', 'test']
    ds = tfds.load('yelp_polarity_reviews', split=data_type, shuffle_files=True)
    for i in ds:
        i = tfds.as_dataframe(i)
        print(i['label'][0])
        csv_path = os.path.join(dir_path, f'yelp_polarity_reviews_{i["label"][0]}.csv')
        i.to_csv(csv_path, index=False)


def create_record(row):
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

        new_obs.append({'sentence_id': 0, 'words': text_obs, 'labels': new_lab})
    return new_obs


def threadingPart(df):
    print('start threading')
    _all_records = []
    for i in df[0]:
        records = create_record(i)
        _all_records.extend(records)
    print('end threading')
    return _all_records

def create_rpunct_dataset(orig_data):
    print('create dataset')
    df = pd.DataFrame()
    input = ''
    with open(orig_data) as f:
        input=f.read()
    inputA = input.split('\n')
    df = pd.DataFrame(inputA)
    # chunk_size = int(df.shape[0]/prs)
    # chunks = [df.iloc[df.index[i:i+chunk_size]] for i in range(0, df.shape[0], chunk_size)]
    all_records = []
    for i in df[0]:
        records = create_record(i)
        all_records.extend(records)
    
    # with Pool(prs) as p:
    #     all_records = p.map(threadingPart, chunks)

    print(f"Dataframe samples: {df.shape}")

    return all_records
    # with open(rpunct_dataset_path, 'w') as fp:
    #     json.dump(all_records, fp)


def create_training_samples(all_records, file_out_nm='train_data', num_splits=5):
    """
    Given a looong list of tokens, splits them into 500 token chunks
    thus creating observations. This is for fine-tuning with simpletransformers
    later on.
    """
    print('create training samples')
    random.seed(1337)
    
    _round = 0
    # all_recs = all_records
    while _round < num_splits:
        all_recs = all_records
        observations = []
        # with open(json_loc_file, 'r') as fp:
        #     all_records = json.load(fp)

        size = len(all_recs) // num_splits
        all_recs = all_recs[size * _round:size * (_round + 1)]
        splits = create_tokenized_obs(all_recs)
        full_data = pd.DataFrame(all_recs)
        del all_recs

        for i in splits:
            data_slice = full_data.iloc[i[0]:i[1], ]
            obi = data_slice.values.tolist()
            obi2 = []
            for a in obi:
                if a:
                    obi2.append(a)

            observations.append(obi2)
        _round += 1
        random.shuffle(observations)
        with open(f'{file_out_nm}_{_round}.txt', 'w') as fp2:
            json.dump(observations, fp2)

        del full_data
        del observations


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
        # print(f'{ix:}')
        if ix == loop_end:
            start = -1
        if i['labels'][-1] == "U" and start == -1:
            start = ix
            end = ix + num_toks
            appends.append((start, end))
            loop_end = start + offset

    return appends

if __name__ == "__main__":
    output_file_names = create_train_datasets()
    print(f"Created following files: {output_file_names}")
