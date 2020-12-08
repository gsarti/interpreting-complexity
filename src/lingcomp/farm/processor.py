import csv
import json
import logging
import os
from pathlib import Path
from typing import List

import numpy as np
from farm.data_handler.processor import InferenceProcessor, Processor, TextClassificationProcessor
from farm.data_handler.samples import Sample
from farm.evaluation.metrics import register_metrics
from farm.modeling.tokenization import tokenize_with_metadata, truncate_sequences
from sklearn.preprocessing import StandardScaler

from lingcomp.farm.features import FeaturesEmbeddingSample
from lingcomp.farm.tokenization import CustomTokenizer
from lingcomp.farm.utils import (
    custom_sample_to_features_text,
    convert_features_to_dataset,
    read_examples_from_file,
    read_examples_from_file_token_level,
    samples_to_features_token_regression,
)
from lingcomp.metrics import classification_metrics, regression_metrics, token_level_regression_metrics
from lingcomp.script_utils import load_dicts_from_file, read_tsv


logger = logging.getLogger(__name__)


class CustomProcessor(Processor):
    """ Uses CustomTokenizer for added model support """

    @classmethod
    def load_from_dir(cls, load_dir):
        """
         Infers the specific type of Processor from a config file (e.g. GNADProcessor) and loads an instance of it.

        :param load_dir: str, directory that contains a 'processor_config.json'
        :return: An instance of a Processor Subclass (e.g. GNADProcessor)
        """
        # read config
        processor_config_file = Path(load_dir) / "processor_config.json"
        config = json.load(open(processor_config_file))
        config["inference"] = True
        # init tokenizer
        if "lower_case" in config.keys():
            logger.warning(
                "Loading tokenizer from deprecated FARM config. "
                "If you used `custom_vocab` or `never_split_chars`, this won't work anymore."
            )
            tokenizer = CustomTokenizer.load(
                load_dir, tokenizer_class=config["tokenizer"], do_lower_case=config["lower_case"]
            )
        else:
            tokenizer = CustomTokenizer.load(load_dir, tokenizer_class=config["tokenizer"])
        # we have to delete the tokenizer string from config, because we pass it as Object
        del config["tokenizer"]
        processor = cls.load(tokenizer=tokenizer, processor_name=config["processor"], **config)
        for task_name, task in config["tasks"].items():
            processor.add_task(
                name=task_name,
                metric=task["metric"],
                label_list=task["label_list"],
                label_column_name=task["label_column_name"],
                text_column_name=task.get("text_column_name", None),
                task_type=task["task_type"],
            )
        if processor is None:
            raise Exception("Processor not found")
        return processor


class InferenceProcessor(InferenceProcessor):
    """
    An InferenceProcessor extension allowing to read files from dict
    """

    def file_to_dicts(self, file: str) -> List[dict]:
        return load_dicts_from_file(file)
    

    def _sample_to_features(self, sample) -> dict:
        features = custom_sample_to_features_text(
            sample=sample,
            tasks=self.tasks,
            max_seq_len=self.max_seq_len,
            tokenizer=self.tokenizer,
        )
        return features


# TokenRegressionProcessor
class TokenRegressionProcessor(CustomProcessor):
    """
    Used to handle token-level regression
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        metric=None,
        train_filename="train.tsv",
        dev_filename=None,
        test_filename="test.tsv",
        dev_split=0.1,
        delimiter="\t",
        label_column_names=[],
        label_names=[],
        scaler_mean=None,
        scaler_scale=None,
        proxies=None,
        **kwargs,
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param data_dir: The directory in which the train and dev files can be found.
        :type data_dir: str
        :param metric: name of metric that shall be used for evaluation, e.g. "acc" or "f1_macro".
            Alternatively you can also supply a custom function, that takes preds and labels as args and returns a numerical value.
            For using multiple metrics supply them as a list, e.g ["acc", my_custom_metric_fn].
        :type metric: str, function, or list
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: None
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :type dev_split: float
        :param delimiter: Separator used in the input tsv / csv file. German version of Conll03 uses a whitespace. GermEval 2014 is tab separated \t
        :type delimiter: str
        :param label_column_name: name of the column in the input csv/tsv that shall be used as training labels
        :type label_column_name: str
        :param label_name: name for the internal label variable in FARM (only needed to adjust in rare cases)
        :type label_name: str
        :param scaler_mean: Value to substract from the label for normalization
        :type scaler_mean: float
        :param scaler_scale: Value to divide the label by for normalization
        :type scaler_scale: float
        :param proxies: proxy configuration to allow downloads of remote datasets.
                        Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :type proxies: dict
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        """
        # Custom processor attributes
        self.delimiter = delimiter

        super(TokenRegressionProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies,
        )
        if metric is None:
            metric = "token_level_regression_metrics"
            register_metrics(metric, token_level_regression_metrics)
        if label_column_names and label_names:
            for col_name, l_name in zip(label_column_names, label_names):
                self.add_task(
                    name=l_name,
                    metric=metric,
                    label_list=[scaler_mean, scaler_scale],
                    label_column_name=col_name,
                    task_type="token_regression",
                    label_name=l_name,
                )
        else:
            logger.info(
                "Initialized processor without tasks. Supply `label_names` and `label_column_names` to the constructor for "
                "using the default task or add a custom task later via processor.add_task()"
            )

    def _create_dataset(self, keep_baskets=False):
        features_flat = []
        for basket in self.baskets:
            for sample in basket.samples:
                features_flat.extend(sample.features)
        if not keep_baskets:
            # free up some RAM, we don't need baskets from here on
            self.baskets = None
        label_tensor_names = [task["label_tensor_name"] for task in self.tasks.values()]
        dataset, tensor_names = convert_features_to_dataset(
            features=features_flat, label_tensor_names=label_tensor_names
        )
        return dataset, tensor_names

    def file_to_dicts(self, file: str) -> List[dict]:
        column_mappings = {task["label_column_name"]: task["label_name"] for task in self.tasks.values()}
        dicts = read_examples_from_file_token_level(
            filename=file, score_cols=column_mappings, word_col="word", sentenceid_col="sentence_id",
        )
        for task in self.tasks.values():
            train_labels = []
            for d in dicts:
                train_labels += [float(label) for label in d[task["label_name"]]]
            scaler = StandardScaler()
            scaler.fit(np.reshape(train_labels, (-1, 1)))
            task["label_list"] = [scaler.mean_.item(), scaler.scale_.item()]
        return dicts

    def _dict_to_samples(self, dictionary: dict, **kwargs) -> List[Sample]:
        # this tokenization also stores offsets, which helps to map our entity tags back to original positions
        tokenized = tokenize_with_metadata(dictionary["text"], self.tokenizer)
        if len(tokenized["tokens"]) == 0:
            text = dictionary["text"]
            logger.warning(
                f"The following text could not be tokenized, likely because it contains a character that the tokenizer does not recognize: {text}"
            )
            return []
        # truncate tokens, offsets and start_of_word to max_seq_len that can be handled by the model
        for seq_name in tokenized.keys():
            tokenized[seq_name], _, _ = truncate_sequences(
                seq_a=tokenized[seq_name], seq_b=None, tokenizer=self.tokenizer, max_seq_len=self.max_seq_len
            )
        # Samples don't have labels during Inference mode
        for task_name, task in self.tasks.items():
            if task_name in dictionary:
                scaled_dict_labels = []
                for label in dictionary[task_name]:
                    label = float(label)
                    scaled_label = (label - task["label_list"][0]) / task["label_list"][1]
                    scaled_dict_labels.append(scaled_label)
                dictionary[task_name] = scaled_dict_labels
        return [Sample(id=None, clear_text=dictionary, tokenized=tokenized)]

    def _sample_to_features(self, sample) -> dict:
        features = samples_to_features_token_regression(
            sample=sample, tasks=self.tasks, max_seq_len=self.max_seq_len, tokenizer=self.tokenizer,
        )
        return features


class CustomRegressionProcessor(CustomProcessor):
    """
    Used to handle a regression dataset in tab separated text + label
    Generalize the FARM base RegressionProcessor class to work on multitask regression
    Also supports the use of custom input features alongside text
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        metric=None,
        train_filename="train.tsv",
        dev_filename=None,
        test_filename="test.tsv",
        dev_split=0.1,
        delimiter="\t",
        quote_char=csv.QUOTE_NONE,
        skiprows=None,
        label_column_names=[],
        label_names=[],
        scaler_mean=None,
        scaler_scale=None,
        proxies=None,
        start_feat_col=None,
        text_column_name="text",
        **kwargs,
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param data_dir: The directory in which the train and dev files can be found.
        :type data_dir: str
        :param label_list: list of labels to predict (strings). For most cases this should be: ["start_token", "end_token"]
        :type label_list: list
        :param metric: name of metric that shall be used for evaluation, e.g. "acc" or "f1_macro".
                 Alternatively you can also supply a custom function, that takes preds and labels as args and returns a numerical value.
                 For using multiple metrics supply them as a list, e.g ["acc", my_custom_metric_fn].
        :type metric: str, function, or list
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: None
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :type dev_split: float
        :param delimiter: Separator used in the input tsv / csv file
        :type delimiter: str
        :param quote_char: Character used for quoting strings in the input tsv/ csv file
        :type quote_char: str
        :param skiprows: number of rows to skip in the tsvs (e.g. for multirow headers)
        :type skiprows: int
        :param label_column_name: name of the column in the input csv/tsv that shall be used as training labels
        :type label_column_name: str
        :param label_name: name for the internal label variable in FARM (only needed to adjust in rare cases)
        :type label_name: str
        :param scaler_mean: Value to substract from the label for normalization
        :type scaler_mean: float
        :param scaler_scale: Value to divide the label by for normalization
        :type scaler_scale: float
        :param proxies: proxy configuration to allow downloads of remote datasets.
                        Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :type proxies: dict
        :param text_column_name: name of the column in the input csv/tsv that shall be used as training text
        :type text_column_name: str
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        """

        # Custom processor attributes
        self.delimiter = delimiter
        self.quote_char = quote_char
        self.skiprows = skiprows
        self.text_column_name = text_column_name
        self.features = start_feat_col
        self.feat_size = None

        super(CustomRegressionProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies,
        )
        if metric is None:
            metric = "regression_metrics"
            register_metrics(metric, regression_metrics)
        if label_column_names and label_names:
            for col_name, l_name in zip(label_column_names, label_names):
                self.add_task(
                    name=l_name,
                    metric=metric,
                    label_list=[scaler_mean, scaler_scale],
                    label_column_name=col_name,
                    task_type="regression",
                    label_name=l_name,
                )

    def file_to_dicts(self, file: str) -> List[dict]:
        dicts = read_examples_from_file(
            file,
            [task["label_name"] for task in self.tasks.values()],
            self.text_column_name,
            start_feat_col=self.features,
        )
        if self.features:
            self.feat_size = len(dicts[0]["features"])
        # collect all labels and compute scaling stats
        for task in self.tasks.values():
            train_labels = [float(d[task["label_name"]]) for d in dicts]
            scaler = StandardScaler()
            scaler.fit(np.reshape(train_labels, (-1, 1)))
            # add to label list in regression task
            task["label_list"] = [scaler.mean_.item(), scaler.scale_.item()]
        return dicts

    def _dict_to_samples(self, dictionary: dict, **kwargs) -> List[Sample]:
        # this tokenization also stores offsets
        tokenized = tokenize_with_metadata(dictionary["text"], self.tokenizer)
        if len(tokenized["tokens"]) == 0:
            text = dictionary["text"]
            logger.warning(
                f"The following text could not be tokenized, likely because it contains a character that the tokenizer does not recognize: {text}"
            )
            return []
        # truncate tokens, offsets and start_of_word to max_seq_len that can be handled by the model
        for seq_name in tokenized.keys():
            tokenized[seq_name], _, _ = truncate_sequences(
                seq_a=tokenized[seq_name], seq_b=None, tokenizer=self.tokenizer, max_seq_len=self.max_seq_len
            )
        # Samples don't have labels during Inference mode
        for task_name, task in self.tasks.items():
            if task_name in dictionary:
                label = float(dictionary[task_name])
                scaled_label = (label - task["label_list"][0]) / task["label_list"][1]
                dictionary[task_name] = scaled_label
        if self.features:
            feats_embed = dictionary.pop("features")
            return [
                FeaturesEmbeddingSample(id=None, clear_text=dictionary, tokenized=tokenized, feat_embeds=feats_embed)
            ]
        return [Sample(id=None, clear_text=dictionary, tokenized=tokenized)]

    def _sample_to_features(self, sample) -> dict:
        features = custom_sample_to_features_text(
            sample=sample, tasks=self.tasks, max_seq_len=self.max_seq_len, tokenizer=self.tokenizer
        )
        if self.features:
            features[0]["feats"] = list(sample.feats_embed)
        return features

    # Custom conversion function is fundamental to avoid rounding
    def _create_dataset(self, keep_baskets=False):
        features_flat = []
        for basket in self.baskets:
            for sample in basket.samples:
                features_flat.extend(sample.features)
        if not keep_baskets:
            # free up some RAM, we don't need baskets from here on
            self.baskets = None
        label_tensor_names = [task["label_tensor_name"] for task in self.tasks.values()]
        if self.features:
            label_tensor_names += ["feats"]
        dataset, tensor_names = convert_features_to_dataset(
            features=features_flat, label_tensor_names=label_tensor_names
        )
        return dataset, tensor_names


class CustomClassificationProcessor(TextClassificationProcessor):
    def __init__(
        self,
        tokenizer,
        max_seq_len,
        data_dir,
        label_list=None,
        metric=None,
        train_filename="train.tsv",
        dev_filename=None,
        test_filename="test.tsv",
        dev_split=0.1,
        delimiter="\t",
        quote_char=csv.QUOTE_NONE,
        skiprows=None,
        label_column_names=[],
        label_names=[],
        multilabel=False,
        header=0,
        proxies=None,
        max_samples=None,
        text_column_name="text",
        **kwargs,
    ):
        self.delimiter = delimiter
        self.quote_char = quote_char
        self.skiprows = skiprows
        self.header = header
        self.max_samples = max_samples
        self.text_column_name = text_column_name

        super(TextClassificationProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies,
        )
        if metric is None:
            metric = "classification_metrics"
            register_metrics(metric, classification_metrics)
        if multilabel:
            task_type = 'multilabel_classification'
        else:
            task_type = "classification"
        data = read_tsv(os.path.join(data_dir, train_filename))
        if label_column_names and label_names:
            for col_name, l_name in zip(label_column_names, label_names):
                self.add_task(
                    name=l_name,
                    metric=metric,
                    label_list=list(set(data[col_name])),
                    label_column_name=col_name,
                    task_type=task_type,
                    label_name=l_name,
                )

    def file_to_dicts(self, file: str) -> List[dict]:
        dicts = read_examples_from_file(
            file, [task["label_name"] for task in self.tasks.values()], self.text_column_name,
        )
        return dicts
    
    def _sample_to_features(self, sample) -> dict:
        features = custom_sample_to_features_text(
            sample=sample,
            tasks=self.tasks,
            max_seq_len=self.max_seq_len,
            tokenizer=self.tokenizer,
        )
        return features
