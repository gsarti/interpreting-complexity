# fmt: off
# flake8: noqa 
"""
    Call as python scripts/shortcuts/compute_similarity.py \
	        --models models/model_a models/model_b models/model_c
    To compute PWCCA and RSA (both inter and intra-model) on all model combinations
    for all layers, using CLS, reduce_mean and per_token strategies.
    isort:skip_file
"""

import argparse
import json
import logging
import os
import sys
from itertools import combinations

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from compute_similarity import main as compute_similarity

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s  %(message)s", datefmt="%d-%m-%y %H:%M:%S", level=logging.INFO,
)

logger = logging.getLogger(__name__)


def compute_similarity_all(args):
    logger.info(f"Script parameters:{json.dumps(vars(args), indent=1)}")
    for method, save_dir in zip(args.similarity_methods, args.save_dirs):
        for model_a, model_b in combinations(args.models, 2):
            for strategy in args.strategies:
                params = [
                    "--similarity_function", method,
                    "--data_path", args.data_path,
                    "--save_dir", save_dir,
                    "--model_a", model_a,
                    "--model_b", model_b,
                    "--strategy", strategy,
                    "--start_layer", f"{args.start_layer}",
                    "--end_layer", f"{args.end_layer}",
                    "--cache_acts",
                    "--save_results",
                    "--avg_pseudodist"
                ]
                logger.info(f"Computing {model_a} vs. {model_b} {method} using {strategy}")
                compute_similarity(params)
        for model in args.models:
            for strategy in args.strategies:
                params = [
                    "--similarity_function", method,
                    "--data_path", args.data_path,
                    "--save_dir", save_dir,
                    "--model_a", model,
                    "--model_b", model,
                    "--strategy", strategy,
                    "--start_layer", f"{args.start_layer}",
                    "--end_layer", f"{args.end_layer}",
                    "--cache_acts",
                    "--save_results",
                    "--avg_pseudodist"
                ]
                logger.info(f"Computing {model} vs. {model} {method} using {strategy}")
                compute_similarity(params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/eval/sst.tsv",
        help="Path to the file containing test input. Default: %(default)s)",
    )
    parser.add_argument(
        "--save_dirs",
        nargs="+",
        default=["logs/rsa_scores", "logs/pwcca_scores"],
        help="The directories where results must be saved. Default: %(default)s)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Full path from root project folder for the models that should be compared with similarity measures."
        "All model combinations are computed.",
    )
    parser.add_argument(
        "--similarity_methods",
        nargs="+",
        choices=["RSA", "PWCCA"],
        default=["RSA", "PWCCA"],
        help="Similarity methods used to compare models. Default: %(default)s, choices: %(choice)s",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["cls_token", "reduce_mean", "per_token"],
        default=["cls_token", "reduce_mean", "per_token"],
        help="Strategies that must be tested for each model combination. Default: %(default)s, choices: %(choice)s",
    )
    parser.add_argument(
        "--start_layer",
        type=int,
        default=-1,
        help="Starting layer for which the similarity should be computed. Default -1 (last layer)"
        "Accepted values are in the range of layers (-1 to -12 for base models)"
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--end_layer",
        type=int,
        default=-12,
        help="Ending layer for which the similarity should be computed. Default -1 (last layer)"
        "Accepted values are in the range of layers (-1 to -12 for base models)"
        "(default: %(default)s)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--cache_activations",
        dest="cache_activations",
        action="store_true",
        help="Save models activations inside pickled files.",
    )
    group.add_argument("--no_cache_activations", dest="cache_activations", action="store_false")
    group_b = parser.add_mutually_exclusive_group()
    group_b.add_argument(
        "--save_results",
        dest="save_results",
        action="store_true",
        help="Set this flag if you want to save results to a tsv file.",
    )
    group_b.add_argument("--no_save_results", dest="save_results", action="store_false")
    parser.set_defaults(cache_activations=True, save_results=True, inter_layer_rsa=True)
    args = parser.parse_args()
    compute_similarity_all(args)


if __name__ == "__main__":
    main()
