import logging
import os
import sys

import numpy as np
import pandas as pd
from transformers import HfArgumentParser

from lingcomp.args import SimilarityArguments
from lingcomp.plotting_utils import create_heatmap, create_lineplot
from lingcomp.similarity import (
    SimilarityFunction,
    SimilarityPlotMode,
    build_distance_matrix,
    build_similarity_df,
    compute_pwcca,
    compute_rsa,
    get_layers,
    get_model_activations,
    get_model_names,
)


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s  %(message)s", datefmt="%d-%m-%y %H:%M:%S", level=logging.INFO,
)

logger = logging.getLogger(__name__)


def get_similarity_value(args, acts_a, acts_b):
    if args.similarity_function == SimilarityFunction.PWCCA:
        # See tutorials at https://github.com/google/svcca for more info
        pwcca_mean, _, _ = compute_pwcca(acts_a, acts_b, epsilon=1e-10)
        if args.avg_pseudodist:
            logger.info("Averaging pseudodistances for both possible orders")
            pwcca_mean_rev, _, _ = compute_pwcca(acts_b, acts_a, epsilon=1e-10)
            pwcca_mean = (pwcca_mean + pwcca_mean_rev) / 2
        score = 1 - pwcca_mean
    elif args.similarity_function == SimilarityFunction.RSA:
        # See tutorial at https://samiraabnar.github.io/articles/2020-05/vizualization for more info
        score = compute_rsa(acts_a, acts_b, f_sim="corr", max_obs=50000)
    else:
        raise AttributeError(f"{args.similarity_function.value} is not defined as similarity function.")
    del acts_a
    del acts_b
    return score


def compute_similarity_scores(args):
    scores = []
    layers_a, layers_b = [], []
    layers = get_layers(args.start_layer, args.end_layer)
    if args.model_a != args.model_b:  # Similarity across the same layer of different models
        for layer in layers:
            acts_a = get_model_activations(
                args.data_path, args.model_a, layer, args.batch_size, args.strategy, args.cache_acts,
            )
            acts_b = get_model_activations(
                args.data_path, args.model_b, layer, args.batch_size, args.strategy, args.cache_acts,
            )
            score = get_similarity_value(args, acts_a, acts_b)
            scores.append(score)
            logger.info(f"{args.similarity_function.value} score at layer {layer}: {score}")
            # Avoid memory management problems when calling this in a loop
        logger.info("Summary:")
        for i, layer in enumerate(layers):
            logger.info(f"Layer {layer}: {scores[i]}")
    elif args.start_layer != args.end_layer:  # Similarity across layers of same model
        for layer_a in layers:
            for layer_b in [x for x in range(layer_a - 1, args.end_layer - 1, -1)]:
                acts_a = get_model_activations(
                    args.data_path, args.model_a, layer_a, args.batch_size, args.strategy, args.cache_acts,
                )
                acts_b = get_model_activations(
                    args.data_path, args.model_b, layer_b, args.batch_size, args.strategy, args.cache_acts,
                )
                score = get_similarity_value(args, acts_a, acts_b)
                scores.append(score)
                layers_a.append(layer_a)
                layers_b.append(layer_b)
                logger.info(f"{args.similarity_function.value} score at layers {layer_a} & {layer_b}: {score}")
    else:
        raise TypeError("Either the models should be different, or more than one layer must be specified.")
    if args.save_results:
        name_a = args.model_a.split("/")[-1]
        name_b = args.model_b.split("/")[-1]
        out = os.path.join(
            args.save_dir,
            f"{args.similarity_function.value}_{args.data_name}_{name_a}_{name_b}_{args.strategy.value}.tsv",
        )
        if args.model_a != args.model_b:
            df = pd.DataFrame({"Layer": layers, f"{args.similarity_function.value} Score": scores})
        else:
            df = pd.DataFrame(
                {"Layer A": layers_a, "Layer B": layers_b, f"{args.similarity_function.value} Score": scores}
            )
        df.to_csv(out, sep="\t", index=False)


def plot_similarity(args):
    logger.info(
        f'Plotting results for strategy "{args.strategy.value}" on data "{args.data_path}"'
        f'using mode "{args.plot_mode.value}"'
    )
    model_names = get_model_names(args.save_dir, args.strategy, args.data_name, use_aliases=args.use_aliases)
    df = build_similarity_df(args.save_dir, args.strategy, args.data_name, model_names, args.plot_mode)
    try:
        if args.plot_mode == SimilarityPlotMode.INTRA:
            for model in list(model_names.values()):
                m_df = df[df["Model A"] == model]
                p_df = m_df.pivot("Layer A", "Layer B", f"{args.similarity_function.value} Score")
                p_df = (
                    p_df.reindex(sorted(p_df.index, reverse=True), axis=0)
                    .reindex(sorted(p_df.columns, reverse=True), axis=1)
                    .round(2)
                )
                
                mask = np.tril(np.ones_like(p_df, dtype=np.bool), k=-1)
                plot_name = f"{args.plot_mode.value}_{model}_{args.data_name}_{args.strategy.value}.png"
                plot_path = os.path.join(args.plot_dir, plot_name)
                create_heatmap(p_df, plot_path, mask=mask)
        elif args.plot_mode == SimilarityPlotMode.INTER:
            plot_name = f"{args.plot_mode.value}_{args.data_name}_{args.strategy.value}.png"
            plot_path = os.path.join(args.plot_dir, plot_name)
            create_lineplot(
                df, plot_path, y=f"{args.similarity_function.value} Score", x="Layer", hue="Combination",
            )
            for layer in get_layers(args.start_layer, args.end_layer):
                ldf = df[df["Layer"] == layer]
                mat = build_distance_matrix(
                    ldf, "Model A", "Model B", f"{args.similarity_function.value} Score", diag_val=1
                )
                logger.info(f"{mat}")
                mask = np.tril(np.ones_like(mat, dtype=np.bool), k=-1)
                plot_name = f"{args.plot_mode.value}_{layer}_{args.data_name}_{args.strategy.value}.png"
                plot_path = os.path.join(args.plot_dir, plot_name)
                create_heatmap(mat, plot_path, mask)
    except:
        raise AttributeError(
            "Check that the argument save_dir corresponds to the actual folder"
            " containing dataframes that should be used for plotting."
        )


def main(params):
    parser = HfArgumentParser(SimilarityArguments)
    args = parser.parse_args_into_dataclasses(params)[0]
    logger.info(f"Script parameters:{args.to_json_string()}")
    if args.plot:
        plot_similarity(args)
    else:
        compute_similarity_scores(args)


if __name__ == "__main__":
    main(sys.argv[1:])
