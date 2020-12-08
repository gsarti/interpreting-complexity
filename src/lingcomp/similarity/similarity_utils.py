import logging
import os
import pickle

import numpy as np
import pandas as pd
from farm.infer import Inferencer
from farm.modeling.tokenization import Tokenizer

from lingcomp.farm.processor import InferenceProcessor
from lingcomp.similarity import SimilarityPlotMode, SimilarityStrategy


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s  %(message)s", datefmt="%d-%m-%y %H:%M:%S", level=logging.INFO,
)

logger = logging.getLogger(__name__)


def get_inferencer(model_name_or_path, batch_size, strategy, layer):
    # Load inferencers for the two models
    model = Inferencer.load(
        model_name_or_path,
        batch_size=batch_size,
        gpu=True,
        task_type="embeddings",
        extraction_strategy=strategy.value,
        extraction_layer=layer,
    )
    if os.path.exists(model_name_or_path):
        model.processor = InferenceProcessor.load_from_dir(model_name_or_path)
        from_model_hub = False
    else:  # Convert processor in our custom InferenceProcessor to be able to load from file
        # This is to have casing, default for saved Albert models and especially needed
        # to match the shape of embeddings when using per_token strategy
        # keep_accents is True by default in FARM
        tokenizer = Tokenizer.load(model_name_or_path, do_lower_case=False, max_len=512,)
        model.processor = InferenceProcessor(tokenizer, model.processor.max_seq_len)
        from_model_hub = True
    return model, from_model_hub


def create_activations_for_layer(model, data_path, strategy, max_size=50000):
    results = model.inference_from_file(data_path)
    # Stack activations of the two models
    if strategy == SimilarityStrategy.PER_TOKEN:
        # Here vecs for each sentence have shape (n_tokens, embed_dim)
        # where n_tokens are the tokens for each sequence
        # Compared models MUST use the same tokenizer in this case to
        # avoid shape mismatches!
        # Out shape = (n_unpadded_tokens_per_sent * n_sentences, embed_dim)
        activations = np.vstack(tuple([x["vec"] for x in results])).T
    else:
        # With other strategies vecs are reduced to shape (,embed_dim)
        # Out shape = (n_sentences, embed_dim)
        activations = np.column_stack(tuple([x["vec"] for x in results]))
    return activations[:max_size, :max_size]


def load_or_compute_activations(data_path, acts_path, model_name_or_path, layer, batch_size, strategy, max_size=50000):
    loaded = False
    try:
        logger.info("Trying to load precomputed activations...")
        # Load precomputed embeddings
        pkl_path = os.path.join(acts_path, f"{layer}.pkl")
        # Used to see if some embeddings from a hub model are already stored
        true_path = pkl_path if os.path.exists(pkl_path) else os.path.join("models", pkl_path)
        if os.path.exists(true_path):
            with open(true_path, "rb") as f:
                activations = pickle.load(f)
            loaded, from_model_hub = True, False
            logger.info(f"Loaded from {true_path}")
        else:
            logger.info(f"Activations not found in path {pkl_path}. Computing them instead...")
            model, from_model_hub = get_inferencer(model_name_or_path, batch_size, strategy, layer)
            activations = create_activations_for_layer(model, data_path, strategy, max_size)
        return activations, loaded, from_model_hub
    except:
        logger.info("Problem with activation loading. Computing them instead...")
        model, from_model_hub = get_inferencer(model_name_or_path, batch_size, strategy, layer)
        activations = create_activations_for_layer(model, data_path, strategy, max_size)
        return activations, loaded, from_model_hub


def get_model_activations(
    data_path, model_name, layer, batch_size, strategy, do_cache_activations=True, max_size=50000
):
    # Used to load or cache embeddings
    data_name = data_path.split("/")[-1].split(".")[0]
    acts_path = os.path.join(model_name, data_name, strategy.value)
    # We use load to decide if we need to cache activations again
    acts, load, from_model_hub = load_or_compute_activations(
        data_path, acts_path, model_name, layer, batch_size, strategy, max_size,
    )
    if do_cache_activations and not load:
        cache_activations(acts, acts_path, layer, from_model_hub)
    logger.info(f"Model {model_name} activation shape: {acts.shape}")
    return acts


def cache_activations(activations, path, layer, from_model_hub):
    if from_model_hub:
        path = os.path.join("models", path)
    logger.info(f"Caching activations in path {path}...")
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, f"{layer}.pkl"), "wb") as f:
        pickle.dump(activations, f)


def build_similarity_df(save_dir, strategy, data_name, model_names, plot_mode):
    combinations = []
    for filename in os.listdir(save_dir):
        if filename.endswith(f"{strategy.value}.tsv") and filename.split("_")[1] == data_name:
            df = pd.read_csv(os.path.join(save_dir, filename), sep="\t")
            mtype_a = model_names[filename.split("_")[2]]
            mtype_b = model_names[filename.split("_")[3]]
            if (mtype_a == mtype_b and plot_mode == SimilarityPlotMode.INTRA) or (
                mtype_a != mtype_b and plot_mode == SimilarityPlotMode.INTER
            ):
                df["Model A"] = mtype_a
                df["Model B"] = mtype_b
                df["Combination"] = f"{mtype_a} <=> {mtype_b}"
                combinations.append(df)
    df = pd.concat(combinations)
    df.sort_values(by=["Combination"], inplace=True)
    return df


def build_distance_matrix(df, index, columns, values, diag_val=0):
    df_symm = df.copy()
    df_symm[index], df_symm[columns] = df[columns], df[index]
    df = pd.concat([df, df_symm], ignore_index=True)
    mat = df.pivot(index=index, columns=columns, values=values)
    mat.values[[np.arange(mat.shape[0])] * 2] = diag_val
    # Sort alphabetically
    mat = mat.reindex(sorted(mat.columns), axis=1).round(2)
    return mat


def get_model_names(dir, strategy, data_name, use_aliases=False):
    models_a = [
        f.split("_")[2]
        for f in os.listdir(dir)
        if f.endswith(f"{strategy.value}.tsv") and f.split("_")[1] == data_name
    ]
    models_b = [
        f.split("_")[3]
        for f in os.listdir(dir)
        if f.endswith(f"{strategy.value}.tsv") and f.split("_")[1] == data_name
    ]
    model_aliases = {}
    for model in set(models_a + models_b):
        if use_aliases:
            model_aliases[model] = input(f"Provide an alias for model {model}:")
        else:
            model_aliases[model] = model
    return model_aliases


def get_layers(start, end):
    return [start] if start == end else [x for x in range(start, end - 1, -1)]
