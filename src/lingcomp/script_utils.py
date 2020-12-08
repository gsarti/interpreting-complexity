import csv
import json
import logging
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from random import shuffle

import numpy as np
import pandas as pd
import torch


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s  %(message)s", datefmt="%d-%m-%y %H:%M:%S", level=logging.INFO,
)

logger = logging.getLogger(__name__)


def compute_agreement(data, mean_vec, std_vec):
    agreement = []
    for idx, row in data.iterrows():
        count = 0
        up_bound = mean_vec[idx] + std_vec[idx]
        low_bound = mean_vec[idx] - std_vec[idx]
        for score in row:
            if score < up_bound and score > low_bound:
                count += 1
        agreement.append(count)
    return agreement


def train_test_split_sentences(data, test_frac=0.1, sentenceid_col="sentence_id"):
    """ Used to perform train-test split without separating sentences, using word-level data """
    group = data.groupby(sentenceid_col)
    sentences = [group.get_group(key) for key, _ in group]
    shuffle(sentences)
    test_len = int(len(sentences) * (1 - test_frac))
    test = sentences[test_len:]
    train = sentences[:test_len]
    train = train[0].append(train[1:])
    test = test[0].append(test[1:])
    return train, test


def reindex_sentence_df(data, sentenceid_col="sentence_id"):
    """ Used for reindexing sentence ids after merging multiple datasets """
    new_idx = []
    curr_sent_id, i = -1, 0
    for _, r in data.iterrows():
        if r[sentenceid_col] != curr_sent_id:
            i += 1
            curr_sent_id = r[sentenceid_col]
        new_idx.append(i)
    data[sentenceid_col] = new_idx
    return data


def save_tsv(data, fname):
    data.to_csv(fname, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")


def read_tsv(fname):
    return pd.read_csv(fname, sep="\t", quoting=csv.QUOTE_NONE, escapechar="\\", engine="python")


def get_labels(data_path, label_column):
    labels = set()
    for data_file in os.listdir(data_path):
        if not data_file.endswith(".tsv"):
            continue
        data = read_tsv(os.path.join(data_path, data_file))
        labels.update(set(data[label_column]))
    return list(labels)


def compute_weighted_loss(task_weights, label_columns):
    if len(task_weights) != len(label_columns):
        raise AttributeError(
            "When specified, task weights should match the number of label columns"
            "a.k.a. len(task_weights) == len(label_columns)"
        )

    def weighted_loss(loss_per_head, global_step=None, batch=None):
        """
        We weigth each head loss tensor by its corresponding coefficient
        """
        return sum([x * task_weights[i] for i, x in enumerate(loss_per_head)])

    logger.info("Using weighted loss with following weights:")
    for t, w in zip(label_columns, task_weights):
        logger.info(f"{t}: {w}")
    return weighted_loss


def load_dicts_from_file(fname):
    if not (os.path.exists(fname)):
        logger.info(f"Couldn't find {fname}")
    df = read_tsv(fname)
    raw_dict = df.to_dict(orient="records")
    return raw_dict


def set_seed(seed, cuda=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def apply_parallel(df_group, func, columns):
    f = partial(func, columns)
    with Pool(cpu_count()) as p:
        ret_list = p.map(f, [group for name, group in df_group])
    return pd.concat(ret_list)


# Taken from https://github.com/cpllab/syntaxgym-core/blob/develop/syntaxgym/get_sentences.py
def get_sentences_from_json(path):
    with open(path, "r") as f:
        in_data = json.load(f)
    return get_sentences(in_data)


def get_sentences(in_data):
    """ Custom redefinition to support Python 3.6, load sentences from a SyntaxGym Suite item """
    sentences = []
    for item in in_data["items"]:
        for cond in item["conditions"]:
            regions = [region["content"].lstrip() for region in cond["regions"] if region["content"].strip() != ""]
            sentence = " ".join(regions)
            sentences.append(sentence)
    return sentences
