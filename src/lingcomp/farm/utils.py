import logging
from copy import copy

import torch
from farm.data_handler.utils import expand_labels, pad
from farm.modeling.tokenization import insert_at_special_tokens_pos
from torch.utils.data import TensorDataset

from lingcomp.script_utils import read_tsv


logger = logging.getLogger(__name__)
torch.set_printoptions(threshold=10000)


def read_examples_from_file(filename, score_cols, text_col, max_samples=None, start_feat_col=None):
    """ start_feat_col : Column name of the first feature to be taken into account in the dataset """
    df = read_tsv(filename)
    if max_samples:
        df = df.sample(max_samples)
    columns = [text_col] + score_cols
    df_filter = df[columns]
    raw_dict = df_filter.to_dict(orient="records")
    if start_feat_col:
        logger.info("Reading features from files...")
        for i, row in df.iterrows():
            raw_dict[i]["features"] = row.loc[start_feat_col:].values
    return raw_dict


def read_examples_from_file_token_level(
    filename, score_cols, word_col="word", sentenceid_col="sentence_id",
):
    """Reads data file and creates the list of dict entries.
    Args:
        filename: Name of the data file in TSV format.
        score_cols: Dict of score columns and score names to be retained.
        word_col: Name of the column in the data file where the word is contained.
        sentenceid_col: Name of the column in the data file for the sentence id.
        sep: Separator of the data file
    """
    df = read_tsv(filename)
    grouped_df = df.groupby(sentenceid_col)
    examples = []
    for key, _ in grouped_df:
        example = {}
        group = grouped_df.get_group(key)
        words = list(group[word_col])
        for score_col, task_name in score_cols.items():
            example[task_name] = list(group[score_col])
        example["text"] = " ".join([str(word) for word in words])
        examples.append(example)
    return examples


def samples_to_features_token_regression(sample, tasks, max_seq_len, tokenizer, non_initial_token=-100, **kwargs):
    """
    Generates a dictionary of features for a given input sample that is to be consumed by an token regression model.
    :param sample: Sample object that contains human readable text and score fields from a single token regression data sample
    :type sample: Sample
    :param tasks: A dictionary where the keys are the names of the tasks and the values are the details of the task (e.g. label_list, metric, tensor name)
    :type tasks: dict
    :param max_seq_len: Sequences are truncated after this many tokens
    :type max_seq_len: int
    :param tokenizer: A tokenizer object that can turn string sentences into a list of tokens
    :param non_initial_token: Token that is inserted into the label sequence in positions where there is a
                              non-word-initial token. This is done since the default NER performs prediction
                              only on word initial tokens
    :return: A list with one dictionary containing the keys "input_ids", "padding_mask", "segment_ids", "initial_mask"
             (also "label_ids" if not in inference mode). The values are lists containing those features.
    :rtype: list
    """

    tokens = sample.tokenized["tokens"]
    if tokenizer.is_fast:
        text = sample.clear_text["text"]
        # Here, we tokenize the sample for the second time to get all relevant ids
        # This should change once we git rid of FARM's tokenize_with_metadata()
        inputs = tokenizer(text,
                           return_token_type_ids=True,
                           truncation=True,
                           truncation_strategy="longest_first",
                           max_length=max_seq_len,
                           return_special_tokens_mask=True)

        if (len(inputs["input_ids"]) - inputs["special_tokens_mask"].count(1)) != len(sample.tokenized["tokens"]):
            logger.error(f"FastTokenizer encoded sample {sample.clear_text['text']} to "
                         f"{len(inputs['input_ids']) - inputs['special_tokens_mask'].count(1)} tokens, which differs "
                         f"from number of tokens produced in tokenize_with_metadata().\n"
                         f"Further processing is likely to be wrong!")
    else:
        inputs = tokenizer.encode_plus(
            text=tokens,
            text_pair=None,
            add_special_tokens=True,
            truncation=False,
            return_special_tokens_mask=True,
            return_token_type_ids=True,
            is_split_into_words=False,
        )

    input_ids, segment_ids, special_tokens_mask = (
        inputs["input_ids"],
        inputs["token_type_ids"],
        inputs["special_tokens_mask"],
    )

    # We construct a mask to identify the first token of a word. We will later use it for masked loss and for predicting entities.
    # Special tokens don't count as initial tokens => we add 0 at the positions of special tokens
    # For BERT we add a 0 in the start and end (for CLS and SEP)
    initial_mask = [int(x) for x in sample.tokenized["start_of_word"]]
    initial_mask = insert_at_special_tokens_pos(initial_mask, special_tokens_mask, insert_element=0)
    assert len(initial_mask) == len(input_ids)

    for task_name, task in tasks.items():
        try:
            label_list = task["label_list"]
            label_name = task["label_name"]
            label_tensor_name = task["label_tensor_name"]
            labels_word = sample.clear_text[label_name]
            labels_token = expand_labels(labels_word, initial_mask, non_initial_token)
            label_scores = [float(x) for x in labels_token]
        except ValueError:
            label_scores = None
            problematic_labels = set(labels_token).difference(set(label_list))
            logger.warning(
                f"[Task: {task_name}] Could not convert labels to ids via label_list!"
                f"\nWe found a problem with labels {str(problematic_labels)}"
            )
        except KeyError:
            # For inference mode we don't expect labels
            label_scores = None
            logger.warning(
                f"[Task: {task_name}] Could not convert labels to ids via label_list!"
                "\nIf your are running in *inference* mode: Don't worry!"
                "\nIf you are running in *training* mode: Verify you are supplying a proper label list to your processor and check that labels in input data are correct."
            )

        # This mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        padding_mask = [1] * len(input_ids)

        # Padding up to the sequence length.
        # Normal case: adding multiple 0 to the right
        # Special cases:
        # a) xlnet pads on the left and uses  "4" for padding token_type_ids
        if tokenizer.__class__.__name__ == "XLNetTokenizer":
            pad_on_left = True
            segment_ids = pad(copy(segment_ids), max_seq_len, 4, pad_on_left=pad_on_left)
        else:
            pad_on_left = False
            segment_ids = pad(copy(segment_ids), max_seq_len, 0, pad_on_left=pad_on_left)

        input_ids = pad(copy(input_ids), max_seq_len, tokenizer.pad_token_id, pad_on_left=pad_on_left)
        padding_mask = pad(copy(padding_mask), max_seq_len, 0, pad_on_left=pad_on_left)
        initial_mask = pad(copy(initial_mask), max_seq_len, 0, pad_on_left=pad_on_left)
        if label_scores:
            # Padding token choice here is irrelevant, we just don't consider it for training after masking
            label_scores = pad(label_scores, max_seq_len, 0, pad_on_left=pad_on_left)

        feature_dict = {
            "input_ids": input_ids,
            "padding_mask": padding_mask,
            "segment_ids": segment_ids,
            "initial_mask": initial_mask,
        }

        if label_scores:
            feature_dict[label_tensor_name] = label_scores

    return [feature_dict]


def convert_features_to_dataset(features, label_tensor_names):
    """
    Converts a list of feature dictionaries (one for each sample) into a PyTorch Dataset.

    :param features: A list of dictionaries. Each dictionary corresponds to one sample. Its keys are the
                     names of the type of feature and the keys are the features themselves.
    :param label_tensor_names: Names of label tensors that should be treated as floats.
    :Return: a Pytorch dataset and a list of tensor names.
    """
    tensor_names = list(features[0].keys())
    all_tensors = []
    for t_name in tensor_names:
        if t_name in label_tensor_names:
            cur_tensor = torch.tensor([sample[t_name] for sample in features], dtype=torch.float32)
        else:
            try:
                cur_tensor = torch.tensor([sample[t_name] for sample in features], dtype=torch.long)
            except ValueError:
                cur_tensor = torch.tensor([sample[t_name] for sample in features], dtype=torch.float32)

        all_tensors.append(cur_tensor)

    dataset = TensorDataset(*all_tensors)
    return dataset, tensor_names


def format_multitask_preds(preds):
    """
    Input format: list of dicts (one per task, named with task_name), each having a
        'predictions' list containing dictionaries that represent predictions for each sample.
        Prediction score is represented by the field {task_name}_score in each of those dicts.
    Output format: a list of lists of dictionaries, where now each dictionary include scores for all the
        tasks that were included in the input.
    """
    out = []
    score_names = [f"{task['task']}_score" for task in preds[1:]]
    first_task = preds[0]
    for sentence_idx, sentence in enumerate(first_task["predictions"]):
        out_sent = []
        for token_idx, token in enumerate(sentence):
            for task, score in zip(preds[1:], score_names):
                token[score] = task["predictions"][sentence_idx][token_idx][score]
            out_sent.append(token)
        out.append(out_sent)
    return out


def custom_sample_to_features_text(
    sample, tasks, max_seq_len, tokenizer
):
    """
    Use is_split_into_words instead of is_pretokenized for tokenizer.encode_plus
    """

    if tokenizer.is_fast:
        text = sample.clear_text["text"]
        # Here, we tokenize the sample for the second time to get all relevant ids
        # This should change once we git rid of FARM's tokenize_with_metadata()
        inputs = tokenizer(text,
                           return_token_type_ids=True,
                           truncation=True,
                           truncation_strategy="longest_first",
                           max_length=max_seq_len,
                           return_special_tokens_mask=True)

        if (len(inputs["input_ids"]) - inputs["special_tokens_mask"].count(1)) != len(sample.tokenized["tokens"]):
            logger.error(f"FastTokenizer encoded sample {sample.clear_text['text']} to "
                         f"{len(inputs['input_ids']) - inputs['special_tokens_mask'].count(1)} tokens, which differs "
                         f"from number of tokens produced in tokenize_with_metadata(). \n"
                         f"Further processing is likely to be wrong.")
    else:
        # TODO It might be cleaner to adjust the data structure in sample.tokenized
        tokens_a = sample.tokenized["tokens"]
        tokens_b = sample.tokenized.get("tokens_b", None)

        inputs = tokenizer.encode_plus(
            tokens_a,
            tokens_b,
            add_special_tokens=True,
            truncation=False,  # truncation_strategy is deprecated
            return_token_type_ids=True,
            is_split_into_words=False,
        )

    input_ids, segment_ids = inputs["input_ids"], inputs["token_type_ids"]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    padding_mask = [1] * len(input_ids)

    # Padding up to the sequence length.
    # Normal case: adding multiple 0 to the right
    # Special cases:
    # a) xlnet pads on the left and uses  "4"  for padding token_type_ids
    if tokenizer.__class__.__name__ == "XLNetTokenizer":
        pad_on_left = True
        segment_ids = pad(segment_ids, max_seq_len, 4, pad_on_left=pad_on_left)
    else:
        pad_on_left = False
        segment_ids = pad(segment_ids, max_seq_len, 0, pad_on_left=pad_on_left)

    input_ids = pad(input_ids, max_seq_len, tokenizer.pad_token_id, pad_on_left=pad_on_left)
    padding_mask = pad(padding_mask, max_seq_len, 0, pad_on_left=pad_on_left)

    assert len(input_ids) == max_seq_len
    assert len(padding_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len

    feat_dict = {
        "input_ids": input_ids,
        "padding_mask": padding_mask,
        "segment_ids": segment_ids,
    }

    # Add Labels for different tasks
    for task_name, task in tasks.items():
        try:
            label_name = task["label_name"]
            label_raw = sample.clear_text[label_name]
            label_list = task["label_list"]
            if task["task_type"] == "classification":
                # id of label
                try:
                    label_ids = [label_list.index(label_raw)]
                except ValueError as e:
                    raise ValueError(f'[Task: {task_name}] Observed label {label_raw} not in defined label_list')
            elif task["task_type"] == "multilabel_classification":
                # multi-hot-format
                label_ids = [0] * len(label_list)
                for l in label_raw.split(","):
                    if l != "":
                        label_ids[label_list.index(l)] = 1
            elif task["task_type"] == "regression":
                label_ids = [float(label_raw)]
            else:
                raise ValueError(task["task_type"])
        except KeyError:
            # For inference mode we don't expect labels
            label_ids = None
        if label_ids is not None:
            feat_dict[task["label_tensor_name"]] = label_ids
    return [feat_dict]


def roll(x, shift, dim=-1, fill_pad=None):
    """
    Shifts a torch tensor x on dimension dim by n, filling empty values with fill_pad.
    """
    if 0 == shift:
        return x
    elif shift < 0:
        shift = -shift
        gap = x.index_select(dim, torch.arange(shift, device=x.device))
        if fill_pad is not None:
            gap = fill_pad * torch.ones_like(gap, device=x.device)
        return torch.cat([x.index_select(dim, torch.arange(shift, x.size(dim), device=x.device)), gap], dim=dim)
    else:
        shift = x.size(dim) - shift
        gap = x.index_select(dim, torch.arange(shift, x.size(dim), device=x.device))
        if fill_pad is not None:
            gap = fill_pad * torch.ones_like(gap, device=x.device)
        return torch.cat([gap, x.index_select(dim, torch.arange(shift, device=x.device))], dim=dim)
