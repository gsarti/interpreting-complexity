import argparse
import json
import logging
import os

import numpy as np
from prettytable import PrettyTable
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from lingcomp.script_utils import read_tsv, save_tsv


DEFAULT_CONFIG = {
    "complexity": {
        "path": "data/preprocessed/complexity_data_bin10.tsv",
        "feat_start_column": "n_tokens",
        "task_labels": ["score"],
        "length_bin_feat": "n_tokens",
    },
    "eyetracking": {
        "path": "data/preprocessed/eyetracking_data_sentence_avg_bin10.tsv",
        "feat_start_column": "n_tokens",
        "task_labels": ["fix_count", "first_pass_dur", "tot_fix_dur", "tot_regr_from_dur"],
        "length_bin_feat": "n_tokens",
    },
}
# This feature is the same for all sentences, causes problems in computing correlation
EXCLUDED_FEATURES = ["verbs_gender_dist"]
# Define this as target task for rank comparison
TARGET_TASK = "complexity_score"

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s  %(message)s", datefmt="%d-%m-%y %H:%M:%S", level=logging.INFO,
)

logger = logging.getLogger(__name__)


def compute_corr_ranks(args, data, data_name, curr_config):
    scores = {}
    feats = data.loc[:, curr_config["feat_start_column"] :]
    feat_names = [f for f in feats.columns if f not in EXCLUDED_FEATURES]
    for task in curr_config["task_labels"]:
        # Avoid name clashes across datasets
        task_name = f"{data_name}_{task}"
        scores[task_name] = [(fname, spearmanr(feats[fname].values, data[task].values)) for fname in feat_names]
        if not args.leave_nans:
            scores[task_name] = [s for s in scores[task_name] if not np.isnan(s[1].correlation)]
    return scores


def compute_svr_ranks(args, data, data_name, curr_config):
    scores = {}
    feats = data.loc[:, curr_config["feat_start_column"] :]
    feats.drop(EXCLUDED_FEATURES, axis=1, inplace=True, errors="ignore")
    # Count 0 values in each feature column
    feats_zero = [sum(feats.iloc[:, x].values == 0) for x in range(len(feats.columns))]
    # Minmax data scaling to make coefficient comparable
    scaler = MinMaxScaler()
    feats[feats.columns] = scaler.fit_transform(feats)
    # Mask irrelevant features (nonzero val for < 10% of entries)
    feats_mask = [x < (len(feats) * 0.95) for x in feats_zero]
    for task in curr_config["task_labels"]:
        # Avoid name clashes across datasets
        task_name = f"{data_name}_{task}"
        svr = SVR(kernel="linear")
        svr.fit(feats.values, data[task].values)
        scores[task_name] = [
            (fname, coef) if mask else (fname, 0)
            for fname, coef, mask in zip(feats.columns, np.squeeze(svr.coef_), feats_mask)
        ]
    return scores


def print_ranks(args, ranks, rtype="corr"):
    logger.info(f"Comparing linguistic features importance by tasks by {rtype} scores...")
    out = os.path.join(args.out_dir, f"features_{rtype}_ranks.txt")
    with open(out, args.write_mode) as f:
        for task in ranks.keys():
            f.write(f"### {task} features: ###\n")
            if rtype == "corr":
                table = PrettyTable(["Feature name", "Spearman correlation", "p-value"])
                for feature in ranks[task]:
                    f_name = feature[0]
                    f_corr = round(feature[1].correlation, 2)
                    f_pval = round(feature[1].pvalue, 5)
                    table.add_row([f_name, f_corr, f_pval])
            else:
                table = PrettyTable(["Feature name", "SVR Coefficient", "Position"])
                idx = 1
                for feature in ranks[task]:
                    val = idx if feature[1] > 0 else (-(len(ranks[task]) - idx + 1) if feature[1] < 0 else "N/A")
                    table.add_row([feature[0], feature[1], val])
                    idx += 1
            f.write(str(table))
            f.write("\n\n")


def compare_corr_ranks(args, corr_ranks, target_task):
    """Produces a dictionary having all tasks except for the target task as keys
    Each task item is a list of tuples containing the following information in order:
    diff_corr[task] = [
        (feat_name, rank_diff, target_corr, task_corr, target_pval, task_pval)
    ]"""
    logger.info("Compare feature importance across tasks by subtracting correlation ranks...")
    diff_corr = {}
    non_target_tasks = [t for t in corr_ranks.keys() if t != target_task]
    for task in non_target_tasks:
        diff_corr[task] = []
        task_features = [feat_tup[0] for feat_tup in corr_ranks[task]]
        for idx, feat_tup in enumerate(corr_ranks[target_task], 1):
            feat_name = feat_tup[0]
            # Check that both target and current task contain the feature value
            if feat_name in [x[0] for x in corr_ranks[task]]:
                target_score = feat_tup[1]
                task_score = corr_ranks[task][task_features.index(feat_name)][1]
                # +1 since we start enumerating at 1
                pos_diff = idx - task_features.index(feat_name) + 1
                diff_corr[task].append(
                    (
                        feat_name,
                        pos_diff,
                        target_score.correlation,
                        task_score.correlation,
                        target_score.pvalue,
                        task_score.pvalue,
                    )
                )
    return diff_corr


def print_diff_corr_ranks(args, diff_corr, target_task):
    # Write difference in rankings to file
    out = os.path.join(args.out_dir, "compare_corr_ranks.txt")
    with open(out, args.write_mode) as f:
        for task in diff_corr.keys():
            f.write(f"### {target_task}-{task} most correlated features: ###\n")
            table = PrettyTable(
                [
                    "Feature name",
                    "Rank diff.",
                    f"{target_task} correlation",
                    f"{task} correlation",
                    f"{target_task} p-value",
                    f"{task} p-value",
                ]
            )
            for val in diff_corr[task]:
                feature, pos_diff = val[0], val[1]
                target_corr, task_corr = round(val[2], 2), round(val[3], 2)
                target_pval, task_pval = round(val[4], 5), round(val[5], 5)
                table.add_row([feature, pos_diff, target_corr, task_corr, target_pval, task_pval])
            f.write(str(table))
            f.write("\n\n")
    return diff_corr


def rankings_correlation(args, diff_corr, target_task):
    logger.info("Correlating tasks' rankings...")
    out = os.path.join(args.out_dir, "rankings_correlation.txt")
    # We need features in alphabetic order to measure correlation
    for task in diff_corr.keys():
        diff_corr[task].sort(key=lambda tup: tup[0])
    with open(out, args.write_mode) as f:
        f.write("### Correlations Rankings' Correlation: ###\n")
        table = PrettyTable([" "] + [k for k in diff_corr.keys()])
        target_corr_list = []
        for task in diff_corr.keys():
            target_corrs = [e[2] for e in diff_corr[task]]
            task_corrs = [e[3] for e in diff_corr[task]]
            corr = spearmanr(target_corrs, task_corrs)
            target_corr_list.append(f"{round(corr.correlation, 2)}|{round(corr.pvalue, 5)}")
        table.add_row([target_task] + target_corr_list)
        for task_a in [k for k in diff_corr.keys()][:-1]:
            inter_task_corrs = []
            for task_b in diff_corr.keys():
                task_a_corrs = [e[3] for e in diff_corr[task_a]]
                task_b_corrs = [e[3] for e in diff_corr[task_b]]
                corr = spearmanr(task_a_corrs, task_b_corrs)
                inter_task_corrs.append(f"{round(corr.correlation, 2)}|{round(corr.pvalue, 5)}")
            table.add_row([task_a] + inter_task_corrs)
        f.write(str(table))


def compute_corr_ranks_over_bins(args, config):
    logger.info("Correlate features with task scores over various length bins...")
    # Compute correlation lists for all the length bins
    corr_ranks_per_bin = []
    args.leave_nans = True
    for curr_binsize in range(args.start_bin, args.end_bin + 1, args.bin_step):
        corr_ranks = {}
        for data_name in config.keys():
            data = read_tsv(config[data_name]["path"])
            bin_data = data.loc[
                (data[config[data_name]["length_bin_feat"]] >= curr_binsize - args.bin_width)
                & (data[config[data_name]["length_bin_feat"]] <= curr_binsize + args.bin_width),
                :,
            ]
            logger.info(f"Bin {curr_binsize}±{args.bin_width} examples: {len(bin_data)}")
            if args.save_binned_data:
                name = config[data_name]["path"].split(".")[0] + f"_bin{curr_binsize}.tsv"
                logger.info(f"Saving {curr_binsize}±{args.bin_width} bin to {name}")
                save_tsv(bin_data, name)
            corr_ranks = {**corr_ranks, **(compute_corr_ranks(args, bin_data, data_name, config[data_name]))}
        for task_name in corr_ranks.keys():
            corr_ranks[task_name].sort(key=lambda tup: tup[1].correlation, reverse=True)
        corr_ranks_per_bin.append(corr_ranks)
    # Order first correlation lists by correlation intensity of features
    first_bin_ranks = corr_ranks_per_bin[0]
    for task in first_bin_ranks.keys():
        first_bin_ranks[task].sort(
            key=lambda tup: -1 if np.isnan(tup[1].correlation) else tup[1].correlation, reverse=True
        )
    # Order all correlation lists based on the one for the first bin
    for i in range(len(corr_ranks_per_bin)):
        for task in corr_ranks_per_bin[i].keys():
            corr_ranks_per_bin[i][task].sort(
                key=lambda x: [first_bin_ranks[task].index(tup) for tup in first_bin_ranks[task] if tup[0] == x[0]]
            )
    return corr_ranks_per_bin


def print_corr_ranks_over_bins(args, corr_ranks_per_bin):
    out_path = os.path.join(args.out_dir, f"features_corr_ranks_over_bins_{args.start_bin}_to_{args.end_bin}")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for task in corr_ranks_per_bin[0].keys():
        out = os.path.join(out_path, f"{task}_most_correlated_features_bins_{args.start_bin}_to_{args.end_bin}.tsv")
        with open(out, args.write_mode) as f:
            for curr_binsize in range(args.start_bin, args.end_bin + 1, args.bin_step):
                f.write(f"bin{curr_binsize}±{args.bin_width}\t")
            f.write("feature\n")
            for idx, feature in enumerate(corr_ranks_per_bin[0][task]):
                f_name = feature[0]
                for corr_ranks in corr_ranks_per_bin:
                    f_corr = round(corr_ranks[task][idx][1].correlation, 2)
                    f.write(f"{f_corr}\t")
                f.write(f"{f_name}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Shorthand to perform all analysis steps.")
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the config json file used for linguistic analysis."
        "By default uses the DEFAULT_CONFIG specified in this file.",
    )
    parser.add_argument(
        "--out_dir", type=str, default="logs/feature_analysis", help="Directory in which results will be saved."
    )
    parser.add_argument(
        "--do_feat_corr_ranks", action="store_true", help="Compute correlation ranks between features and task scores."
    )
    parser.add_argument(
        "--do_feat_svr_ranks",
        action="store_true",
        help="Compute SVR coefficient ranks between features and task scores.",
    )
    parser.add_argument("--do_compare_corr_ranks", action="store_true")
    parser.add_argument("--do_rankings_correlation", action="store_true")
    parser.add_argument("--do_feat_corr_ranks_over_bins", action="store_true")
    parser.add_argument(
        "--start_bin",
        type=int,
        default=10,
        help="The starting size bin for which feature correlation should be computed.",
    )
    parser.add_argument(
        "--end_bin", type=int, default=35, help="The ending size bin for which feature correlation should be computed."
    )
    parser.add_argument("--bin_step", type=int, default=5, help="The step size to be taken from start bin to end bin.")
    parser.add_argument(
        "--bin_width",
        type=int,
        default=1,
        help="The +- interval in which scores are considered to be part of the same bin.",
    )
    parser.add_argument(
        "--overwrite_output_files",
        action="store_true",
        help="Specifies that existing output files should be overwritten by new ones."
        "By default, results are appended to existing files.",
    )
    parser.add_argument(
        "--save_binned_data", action="store_true", help="If specified, saves the binned data in tsv format."
    )
    args = parser.parse_args()
    args.leave_nans = False
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if args.config_path is None:
        config = DEFAULT_CONFIG
    else:
        with open(args.config_path, "r") as c:
            config = json.load(c)
    args.write_mode = "w" if args.overwrite_output_files else "a+"
    corr_ranks = {}
    svr_ranks = {}
    if args.all:
        args.do_feat_svr_ranks = True
        args.do_feat_corr_ranks, args.do_compare_corr_ranks = True, True
        args.do_rankings_correlation, args.do_feat_corr_ranks_over_bins = True, True
    for data_name in config.keys():
        data = read_tsv(config[data_name]["path"])
        corr_ranks = {**corr_ranks, **(compute_corr_ranks(args, data, data_name, config[data_name]))}
        if args.do_feat_svr_ranks:
            svr_ranks = {**svr_ranks, **(compute_svr_ranks(args, data, data_name, config[data_name]))}
    for task_name in corr_ranks.keys():
        corr_ranks[task_name].sort(key=lambda tup: tup[1].correlation, reverse=True)
        if args.do_feat_svr_ranks:
            svr_ranks[task_name].sort(key=lambda tup: tup[1], reverse=True)
    if args.do_feat_corr_ranks:
        print_ranks(args, corr_ranks)
    if args.do_feat_svr_ranks:
        print_ranks(args, svr_ranks, rtype="svr")
    if args.do_compare_corr_ranks:
        if len(corr_ranks.keys()) < 2:
            raise AttributeError("At least two tasks should be specified to " "compare correlation ranks.")
        diff_corr = compare_corr_ranks(args, corr_ranks, TARGET_TASK)
        for task_name in diff_corr.keys():
            diff_corr[task_name].sort(key=lambda tup: abs(tup[1]), reverse=True)
        print_diff_corr_ranks(args, diff_corr, TARGET_TASK)
    if args.do_rankings_correlation:
        if len(corr_ranks.keys()) < 2:
            raise AttributeError("At least two tasks should be specified to " "compare correlation ranks.")
        if not args.do_compare_corr_ranks:
            raise AttributeError("Correlation rank differences should be computed to correlate them.")
        rankings_correlation(args, diff_corr, TARGET_TASK)
    if args.do_feat_corr_ranks_over_bins:
        if args.start_bin is None or args.end_bin is None:
            raise AttributeError(
                "start_bin and end_bin argument should be specified " "for feature_corr_ranks_over_bins option."
            )
        ranks_per_bin = compute_corr_ranks_over_bins(args, config)
        print_corr_ranks_over_bins(args, ranks_per_bin)


if __name__ == "__main__":
    main()
