# Using SyntaxGym classes to estimate how eye-tracking metrics training affects language models

import argparse
import logging
import os

from lingcomp.metrics import confidence_intervals
from lingcomp.script_utils import save_tsv
from lingcomp.syntaxgym import compute_suite_et_metrics, evaluate_suite


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s  %(message)s", datefmt="%d-%m-%y %H:%M:%S", level=logging.INFO,
)

logger = logging.getLogger(__name__)


def evaluate_model_on_suite(model_name_or_path, suite_path, save_path=None, conf_interval=0.95):
    """
    Given a model or its predictions, computes its performances on a test suite
    based on the formula specified in it
    Args:
        model_name_or_path: A path to a local folder containing model files
            (HuggingFace or FARM format), or a model name from the HuggingFace
            model hub, or a path to a local TSV file containing model predictions.
        suite_path: A path to a local JSON file containing a suite in SyntaxGym format.
        save_path: If model_name_or_path is a model, its inferred predictions will be saved
            to this path.
        conf_interval: Float between 0 and 1, confidence interval computed on metric values
    Returns:
        A dataframe containing average scores across items for each condition and region,
        along with confidence bounds, and a dataframe containing success ratios over suite
        formula for each score column and prediction formula. E.g.

        condition_name  region_number  score_name  metric_name  mean  sem  count  region  up_conf  low_conf
        ambig_comma  1  first_fix_dur_score  sum  395  14  24  Start  365  424
        ambig_comma  2  first_fix_dur_score  sum  179   5  24  Verb   167  191
        ambig_comma  4  first_fix_dur_score  sum  228   7  24  NP/Z   213  244
        ambig_comma  5  first_fix_dur_score  sum  158   7  24  Verb   143  173

        prediction_id  prediction_formula  score_column  result
        0  (((5;%ambig_nocomma%) > (5;%ambig_comma%)))        first_fix_dur_score  0.66
        1  (((5;%ambig_nocomma%) > (5;%unambig_nocomma%)))    first_fix_dur_score  0.33
        2  ((((5;%ambig_nocomma%) - (5;%ambig_comma%)) > ...  first_fix_dur_score  0.33
    """
    if os.path.exists(model_name_or_path) and model_name_or_path.endswith(".tsv"):
        pred_suite, df = compute_suite_et_metrics(suite_path, load_path=model_name_or_path)
    else:
        pred_suite, df = compute_suite_et_metrics(suite_path, model=model_name_or_path, save_path=save_path)
    # Averaging predictions across conditions, regions, scores and metric names
    grp = df.groupby(["condition_name", "region_number", "score_name", "metric_name"])
    avg_df = grp["metric_val"].agg(["mean", "sem", "count"]).reset_index()
    avg_df["region"] = [pred_suite.region_names[i - 1] for i in avg_df.region_number]
    # Compute confidence intervals
    avg_df["up_conf"], avg_df["low_conf"] = zip(
        *[confidence_intervals(r, conf_interval) for _, r in avg_df.iterrows()]
    )
    avg_df = avg_df.sort_values(["score_name", "condition_name", "region_number"])
    avg_df = avg_df[
        [
            "condition_name",
            "region_number",
            "region",
            "score_name",
            "metric_name",
            "mean",
            "sem",
            "count",
            "up_conf",
            "low_conf",
        ]
    ]
    pred_df = evaluate_suite(pred_suite)
    res_df = pred_df.groupby(["prediction_id", "prediction_formula", "score_column"]).mean()["result"]
    res_df = res_df.reset_index().sort_values(["score_column", "prediction_id"])
    res_df = res_df[["prediction_id", "result", "score_column", "prediction_formula"]]
    if save_path:
        logger.info(f"Saving dataframes to {save_path}")
        save_tsv(avg_df, f"{save_path}/{pred_suite.meta['name']}_avg.tsv")
        save_tsv(res_df, f"{save_path}/{pred_suite.meta['name']}_res.tsv")
    return avg_df, res_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        type=str,
        help="Path to pre-trained model or shortcut name selected from HuggingFace AutoModels."
        "Alternatively, path to the TSV file containing model precomputed predictions.",
    )
    parser.add_argument(
        "--suite_path",
        required=True,
        type=str,
        help="Path to the JSON file containing the test suite in SyntaxGym format.",
    )
    parser.add_argument(
        "--save_path",
        default=None,
        type=str,
        help="If specified, files containing predictions and eval metrics are saved to this path.",
    )
    parser.add_argument(
        "--ci", default=0.95, type=float, help="Confidence level for confidence intervals over averaged values",
    )
    args = parser.parse_args()
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    evaluate_model_on_suite(args.model_name_or_path, args.suite_path, args.save_path, args.ci)


if __name__ == "__main__":
    main()
