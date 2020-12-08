""" Adapted from ZuCo and ZuCo2 scripts at https://osf.io/hbt7s/, for eye-tracking only """

import logging
import os
import re

import h5py
import numpy as np
import pandas as pd
import scipy.io as io
from tqdm import tqdm


logger = logging.getLogger(__name__)


def is_real_word(word):
    """
    Check if the word is a real word
    :param word:    (str)   word string
    :return:
        is_word (bool)  True if it is a real word
    """
    is_word = re.search("[a-zA-Z0-9]", word)
    return is_word


def load_matlab_string(matlab_extracted_object):
    """
    Converts a string loaded from h5py into a python string
    :param matlab_extracted_object:     (h5py)  matlab string object
    :return:
        extracted_string    (str)   translated string
    """
    extracted_string = "".join(chr(c) for c in matlab_extracted_object)
    return extracted_string


def extract_word_level_data(data_container, word_objects):
    """
    Extracts word level eye-tracking data for a specific sentence
    :param data_container:          (h5py)  Container of the whole data, h5py object
    :param word_objects:            (h5py)  Container of all word data for a specific sentence
    :return:
        word_level_data     (dict)  Contains all word level data indexed by their index number in the sentence,
                                    together with the reading order, indexed by "word_reading_order"
    """
    available_objects = list(word_objects)
    word_level_data = {}
    if isinstance(available_objects[0], str):
        content_data = word_objects["content"]

        if "rawET" in available_objects:
            ffd_data = word_objects["FFD"]
            gd_data = word_objects["GD"]
            gpt_data = word_objects["GPT"]
            trt_data = word_objects["TRT"]
            fxc_data = word_objects["nFixations"]
            assert len(content_data) == len(trt_data), "Incorrect number of data for words"
            zipped_data = zip(content_data, ffd_data, gd_data, gpt_data, trt_data, fxc_data)
            word_idx = 0
            for word_obj, ffd, gd, gpt, trt, fxc in zipped_data:
                word_string = load_matlab_string(data_container[word_obj[0]])
                if is_real_word(word_string):
                    data_dict = {}
                    data_dict["FFD"] = (
                        data_container[ffd[0]][()][0, 0] if len(data_container[ffd[0]][()].shape) == 2 else None
                    )
                    data_dict["GD"] = (
                        data_container[gd[0]][()][0, 0] if len(data_container[gd[0]][()].shape) == 2 else None
                    )
                    data_dict["GPT"] = (
                        data_container[gpt[0]][()][0, 0] if len(data_container[gpt[0]][()].shape) == 2 else None
                    )
                    data_dict["TRT"] = (
                        data_container[trt[0]][()][0, 0] if len(data_container[trt[0]][()].shape) == 2 else None
                    )
                    data_dict["FXC"] = (
                        data_container[fxc[0]][()][0, 0] if len(data_container[fxc[0]][()].shape) == 2 else None
                    )
                    data_dict["word_idx"] = word_idx
                    data_dict["content"] = word_string
                    word_level_data[word_idx] = data_dict
                    word_idx += 1
    return word_level_data


def read_zuco1_mat(data_path):
    dfs = []
    for fname in tqdm(os.listdir(data_path)):
        if fname.endswith(".mat"):
            logger.info(f"Processing {fname}")
            name = os.path.join(data_path, fname)
            participant = name.split("_")[0][-3:]
            task_id = name.split("_")[1].split(".")[0]
            data = io.loadmat(name, squeeze_me=True, struct_as_record=False)["sentenceData"]
            for sent_idx, sentence_data in enumerate(data):
                word_level_data = {}
                # Conditions taken from https://osf.io/r6ewq/, TSR not considered (natural reading only)
                # if (task_id == "NR" and participant == "ZPH") and (50 <= sent_idx <= 99):
                #    continue
                # elif (task_id == "NR" and participant == "ZJS") and (sent_idx <= 49):
                #    continue
                # elif (task_id == "SR" and participant == "ZDN") and ((150 <= sent_idx <= 249) or sent_idx == 399):
                #    continue
                if isinstance(sentence_data.word, float):
                    continue
                for word_idx, word_data in enumerate(sentence_data.word):
                    if is_real_word(word_data.content):
                        data_dict = {}
                        data_dict["FFD"] = word_data.FFD if not isinstance(word_data.FFD, np.ndarray) else np.nan
                        data_dict["GD"] = word_data.GD if not isinstance(word_data.GD, np.ndarray) else np.nan
                        data_dict["GPT"] = word_data.GPT if not isinstance(word_data.GPT, np.ndarray) else np.nan
                        data_dict["TRT"] = word_data.TRT if not isinstance(word_data.TRT, np.ndarray) else np.nan
                        data_dict["FXC"] = (
                            word_data.nFixations if not isinstance(word_data.nFixations, np.ndarray) else np.nan
                        )
                        data_dict["word_idx"] = word_idx
                        data_dict["content"] = word_data.content
                        word_level_data[word_idx] = data_dict
                word_level_data = pd.DataFrame.from_dict(word_level_data, orient="index")
                word_level_data["sent_idx"] = sent_idx
                word_level_data["participant"] = participant
                word_level_data["task_id"] = f"{task_id}1"
                dfs.append(word_level_data)
    df = pd.concat(dfs, ignore_index=True)
    return df


def read_zuco2_mat(data_path):
    dfs = []
    for fname in tqdm(os.listdir(data_path)):
        if fname.endswith(".mat"):
            logger.info(f"Processing {fname}")
            name = os.path.join(data_path, fname)
            participant = name.split("_")[0][-3:]
            task_id = name.split("_")[1].split(".")[0]
            # Exclude YMH due to incomplete data because of dyslexia
            if participant != "YMH":
                f = h5py.File(name, "r")
                sentence_data = f["sentenceData"]
                content_data = sentence_data["content"]
                word_data = sentence_data["word"]
                logger.info(f"{len(content_data)} sentences found")
                for idx, content in enumerate(content_data):
                    wl_data = extract_word_level_data(f, f[word_data[idx][0]])
                    wl_data = pd.DataFrame.from_dict(wl_data, orient="index")
                    wl_data["sent_idx"] = idx
                    wl_data["participant"] = participant
                    wl_data["task_id"] = f"{task_id}2"
                    dfs.append(wl_data)
    df = pd.concat(dfs, ignore_index=True)
    return df
