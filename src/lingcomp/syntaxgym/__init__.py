import logging
import os
import re
from copy import deepcopy

import pandas as pd
from syntaxgym import _load_suite
from syntaxgym import utils as orig_utils
from syntaxgym.get_sentences import get_sentences
from syntaxgym.suite import Region

from lingcomp.farm.multitask import MultitaskInferencer
from lingcomp.script_utils import get_sentences_from_json, read_tsv, save_tsv

from .prediction import custom_load_suite


logger = logging.getLogger(__name__)


def get_et_metrics(sentences, model=None, save_path=None, load_path=None, id="model"):
    if load_path is not None and os.path.exists(load_path):
        logger.info(f"Loading predicted eye-tracking metrics from {load_path}")
        df = read_tsv(load_path)
    else:
        logger.info(f"Inferencing eye-tracking predictions with model {model}")
        # Remove all whitespaces before punctuation, to make sure that format actually
        # matches the one used in eye-tracking files on which the model was trained.
        sentences = (
            [{"text": re.sub(r"\s+([^\w\s])", r"\1", s)} for s in sentences]
            if type(sentences[0]) is str
            else sentences
        )
        model = MultitaskInferencer.load(model, gpu=True, level="token")
        res = model.inference_from_dicts(dicts=sentences)
        for i, sent in enumerate(res):
            for j, tok in enumerate(sent):
                res[i][j]["sentence_id"] = i
                res[i][j]["token_id"] = j
        res = [token for sentence in res for token in sentence]
        df = pd.DataFrame.from_dict(res)
        df["context"] = [c.rstrip() for c in df["context"]]
        if save_path is not None:
            logger.info(f"Saving inferenced predictions to {save_path}")
            save_tsv(df, f"{save_path}/{id}_preds.tsv")
    return df


def aggregate_et_metrics_regions(et_df, suite):
    """ Adapted from aggregate_surprisal SyntaxGym method """
    metrics = suite.meta["metric"]
    if metrics == "all":
        metrics = orig_utils.METRICS.keys()
    else:
        # if only one metric specified, convert to singleton list
        metrics = [metrics] if type(metrics) == str else metrics
    orig_utils.validate_metrics(metrics)
    ret = deepcopy(suite)
    sent_idx = 0
    sent_group = et_df.groupby("sentence_id", sort=False, as_index=False)
    # iterate through surprisal file, matching tokens with regions
    for i_idx, item in enumerate(suite.items):
        for c_idx, cond in enumerate(item["conditions"]):
            curr_grp = list(sent_group.groups.keys())[sent_idx]
            sent_vals = sent_group.get_group(curr_grp).reset_index(drop=True)
            temp_idx, t_idx = 0, 0
            # strip down data a bit
            sent_vals = sent_vals.drop(["start", "end", "context", "token_id", "sentence_id"], axis=1)
            sent_vals_cols = sent_vals.columns
            # initialize Sentence object for current sentence
            regions = [Region(**r) for r in cond["regions"]]
            # iterate through regions in sentence
            for r_idx, region in enumerate(regions):
                vals = {}
                t_idx = temp_idx
                words = re.sub(r"[^\w\s]", "", region.content)
                # Skip region if regions doesn't contain proper words
                if words == "":
                    continue
                for score in list(sent_vals_cols):
                    temp_idx = t_idx
                    for token in words.split(" "):
                        region.token_surprisals.append(sent_vals[score][temp_idx])
                        temp_idx += 1
                    # get dictionary of region-level score values for each metric
                    vals[score] = {m: region.agg_surprisal(m) for m in metrics}
                    region.token_surprisals = []
                # insert all scores into original dict
                # in aggregate surprisal vals is a dict of metrics, score pairs
                # here is a dict of dicts (one per score column), each containing metric, score pairs
                ret.items[i_idx]["conditions"][c_idx]["regions"][r_idx]["metric_value"] = vals
            # update sentence counter
            sent_idx += 1
    # Set this for prediction later
    ret.score_columns = list(sent_vals_cols)
    return ret


def compute_suite_et_metrics(suite, return_df=True, **kwargs):
    """
    Compute per-region eye-tracking metrics on the given suite, provided a whitespace-tokenized
    dataframe of metric predictions.
    Args:
        suite: A path or open file stream to a suite JSON file, or an
            already loaded suite dict.
        return_df: If True, returns suite items in DataFrame format alongside the evaluated suite
        kwargs: Refer to get_et_metrics parameters
    Returns:
        An evaluated test suite dict --- a copy of the data from
        ``suite``, now including per-region eye-tracking data.
        Optionally, evaluated items of the suite in DataFrame format
    """
    suite = custom_load_suite(suite)
    # Convert to sentences
    suite_sentences = get_sentences(suite)
    # Get eye-tracking dataframe
    et_df = get_et_metrics(suite_sentences, id=suite.meta["name"], **kwargs)
    # Aggregate over regions and get result suite
    result = aggregate_et_metrics_regions(et_df, suite)
    if not return_df:
        return result
    # Make a nice dataframe
    result_df = pd.DataFrame.from_records(
        [
            (
                item["item_number"],
                cond["condition_name"],
                reg["content"],
                reg["region_number"],
                score_name,
                metric_name,
                metric_val,
            )
            for item in result.items
            for cond in item["conditions"]
            for reg in cond["regions"]
            if "metric_value" in reg
            for score_name, score_dict in reg["metric_value"].items()
            for metric_name, metric_val in score_dict.items()
        ],
        columns=[
            "item_number",
            "condition_name",
            "content",
            "region_number",
            "score_name",
            "metric_name",
            "metric_val",
        ],
    )
    return result, result_df


def evaluate_suite(suite, return_df=True):
    """
    Evaluate prediction results on the given suite. The suite must contain
    eye-tracking metrics estimates for regions.
    """
    suite = custom_load_suite(suite)
    results = suite.evaluate_predictions()
    if not return_df:
        return suite, results
    # Make a nice dataframe
    results_data = [
        (
            suite.meta["name"],
            pred.idx,
            suite.predictions[pred.idx].__str__().lstrip("Prediction"),
            item_number,
            score,
            result,
        )
        for item_number, scores in results.items()
        for score, preds in scores.items()
        for pred, result in preds.items()
    ]
    return pd.DataFrame(
        results_data, columns=["suite", "prediction_id", "prediction_formula", "item_number", "score_column", "result"]
    )
