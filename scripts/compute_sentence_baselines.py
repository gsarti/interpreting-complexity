"""
Computes the baseline for a dataset given the average score for bin of different
lengths in the training set. This baseline should take in account the implicit
importance of sentence length in complexity measurements.
Use very large bins (e.g. 100) for average baseline over the whole dataset.
"""

import argparse
import logging
import os

from sklearn.model_selection import KFold
from sklearn.svm import SVR
from tqdm import tqdm

from lingcomp.metrics import regression_metrics
from lingcomp.script_utils import read_tsv


def baseline(args, train, test):
    y_test = test[args.score_column].values
    bin_scores = {}
    min_val = 0
    max_val = int(round(max(train["n_tokens"].values) / args.bin_size) * args.bin_size + 1)
    for len_bin in range(min_val, max_val, int(args.bin_size)):
        len_bin_score = train[train["n_tokens"] == len_bin].loc[:, args.score_column].mean()
        bin_scores[len_bin] = len_bin_score
    preds = []
    for _, r in test.iterrows():
        # Manage case in which test sentence exceeds training bins
        n_tok = r["n_tokens"]
        if n_tok > max(bin_scores.keys()):
            n_tok = max(bin_scores.keys())
        preds.append(bin_scores[n_tok])
    return regression_metrics(preds, y_test)


def length_svm(args, train, test):
    X_train = train["n_tokens"].values.reshape(-1, 1)
    X_test = test["n_tokens"].values.reshape(-1, 1)
    y_train, y_test = train[args.score_column].values, test[args.score_column].values
    svr = SVR(kernel="linear")
    preds = svr.fit(X_train, y_train).predict(X_test)
    return regression_metrics(preds, y_test)


def ling_feat_svm(args, train, test):
    X_train = train.iloc[:, args.feat_start_idx :].values
    X_test = test.iloc[:, args.feat_start_idx :].values
    y_train, y_test = train[args.score_column].values, test[args.score_column].values
    svr = SVR(kernel="linear")
    preds = svr.fit(X_train, y_train).predict(X_test)
    return regression_metrics(preds, y_test)


def compute_scores_crossval(args, data, score_func):
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    scores = []
    args.logger.info(f"Computing {args.n_splits}-fold CV for {score_func.__name__}")
    for train_idx, test_idx in tqdm(kf.split(data)):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]
        scores.append(score_func(args, train, test))
    avg_scores = {}
    for metric in scores[0].keys():
        avg_metric = sum([x[metric] for x in scores]) / len([x[metric] for x in scores])
        avg_scores[metric] = avg_metric
    return avg_scores


def log_scores(args, score_func, avg_scores):
    args.logger.info(f"Average {score_func.__name__} scores for {args.path}:")
    for metric in avg_scores.keys():
        args.logger.info(f"avg_{metric}: {avg_scores[metric]}")


def write_scores(args, score_func, avg_scores, write_head=False):
    args.logger.info(f"Writing scores for {score_func.__name__} to {args.tsv_path}.")
    avg_scores["model"] = score_func.__name__
    avg_scores["data"] = args.path.split("/")[-1].split(".")[0]
    avg_scores["count"] = args.data_size
    avg_scores["score_column"] = args.score_column
    with open(args.tsv_path, "a+") as f:
        if write_head:
            for val in avg_scores.keys():
                f.write(f"{val}\t")
            f.write("\n")
        for val in avg_scores.keys():
            score = avg_scores[val] if type(avg_scores[val]) is str else round(avg_scores[val], 2)
            f.write(f"{score}\t")
        f.write("\n")


def compute_sentence_baselines():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Shorthand to perform all baseline evaluations")
    parser.add_argument("--stat", action="store_true", help="Perform evaluation with a statistic baseline.")
    parser.add_argument(
        "--svm_len", action="store_true", help="Perform evaluation with an SVM model based on sentence length."
    )
    parser.add_argument(
        "--svm_all", action="store_true", help="Perform evaluation with an SVM model using all linguistic features."
    )
    parser.add_argument("--path", type=str, required=True, help="Path to the file containing the dataset.")
    parser.add_argument(
        "--text_column", type=str, required=True, help="Name of the column in dataset containing sentences."
    )
    parser.add_argument(
        "--score_column", type=str, required=True, help="Name of the column in dataset containing the score."
    )
    parser.add_argument(
        "--n_splits",
        default=5,
        type=int,
        help="Number of train-test splits that should be used to compute the baseline.",
    )
    parser.add_argument("--bin_size", default=5.0, type=float, help="Bin size to compute baseline.")
    parser.add_argument("--log_dir", default="logs", type=str, help="The log dir. Logs of runs will be saved there.")
    parser.add_argument(
        "--log", action="store_true", help="Set this flag if you want to log the current run to a file."
    )
    parser.add_argument("--write_tsv", action="store_true")
    parser.add_argument("--write_tsv_header", action="store_true")
    parser.add_argument("--tsv_path", default="logs/sentence_baselines.tsv")
    args = parser.parse_args()
    handlers = [logging.StreamHandler()]
    if args.all:
        args.stat, args.svm_len, args.svm_all = True, True, True
    # Setup logging to file
    if args.log:
        name = args.path.split("/")[-1].split(".")[0]
        filehandler = logging.FileHandler(os.path.join(args.log_dir, f"baselines_{name}.log"))
        filehandler.setLevel(logging.INFO)
        handlers.append(filehandler)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
        datefmt="%d-%m-%y %H:%M:%S",
        level=logging.INFO,
        handlers=handlers,
    )
    args.logger = logging.getLogger(__name__)
    args.logger.info(vars(args))
    # Load data
    data = read_tsv(args.path)
    args.data_size = len(data)
    if "n_tokens" not in data.columns:
        raise AttributeError("Run preprocess.py with --do_features option to enable baseline computations.")
    if args.svm_len:
        avg_scores = compute_scores_crossval(args, data, length_svm)
        log_scores(args, length_svm, avg_scores)
        if args.write_tsv:
            write_scores(args, length_svm, avg_scores, write_head=args.write_tsv_header)
    if args.svm_all:
        args.feat_start_idx = data.columns.get_loc("n_tokens")
        avg_scores = compute_scores_crossval(args, data, ling_feat_svm)
        log_scores(args, ling_feat_svm, avg_scores)
        if args.write_tsv:
            write_scores(args, ling_feat_svm, avg_scores)
    if args.stat:
        # Round values to nearest bin
        data["n_tokens"] = [int(round(x / args.bin_size) * args.bin_size) for x in data["n_tokens"]]
        avg_scores = compute_scores_crossval(args, data, baseline)
        log_scores(args, baseline, avg_scores)
        if args.write_tsv:
            write_scores(args, baseline, avg_scores)


if __name__ == "__main__":
    compute_sentence_baselines()
