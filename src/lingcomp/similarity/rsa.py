import logging

import numpy as np
from sklearn.preprocessing import normalize


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s  %(message)s", datefmt="%d-%m-%y %H:%M:%S", level=logging.INFO,
)

logger = logging.getLogger(__name__)


def compute_rsa(acts1, acts2, f_sim="dot", max_obs=None):
    """Taken from https://samiraabnar.github.io/articles/2020-05/vizualization:
    First, we feed a sample set of size n from the validation/test set (e.g. 1000 examples) to the forward pass
    of each model and obtain the representation from its penultimate layer (in fact, this can be any layer).
    Those correspond to acts1 and acts2 in 2D arrays shape

    Next, for each model, we calculate the similarity of the representations of all pairs from the sample set using
    some similarity metric, e.g., dot product (f_sim). This leads to m matrices of size n×n.

    We use the samples similarity matrix associated with each model to compute the similarity between all pairs of models.
    To do this, we compute the dot product (we can use any other measure of similarity as well) of the corresponding rows
    of these two matrices after normalization, and average all the similarity of all rows, which leads to a single scalar.

    Given all possible pairs of models, we then have a model similarity matrix of size m×m
    and we apply a multi dimensional scaling algorithm to embed all the models in a 2D space based on their similarities.
    """
    # Now both have shape [ n_samples x n_samples ], where samples can be sentences (for CLS/reduction) or tokens
    # Use the max_obs param to set the max number of observation and avoid going OOM (esp. for per_token)
    max_obs = len(acts1) if max_obs is None else max_obs
    sim1 = acts1[:max_obs, :].dot(acts1[:max_obs, :].T)
    sim2 = acts2[:max_obs, :].dot(acts2[:max_obs, :].T)

    # Normalize similarity matrices
    sim1 = normalize(sim1)
    sim2 = normalize(sim2)

    # The diagonal of the dot product is equivalent to the individual sum of
    # the scalar product of rows of sim1 and columns of sim2
    if f_sim == "dot":
        sim = (sim1 * sim2).sum(-1)
    # Pearson correlation between rows of the two matrices
    elif f_sim == "corr":
        sim = np.diag(np.corrcoef(sim1, sim2)[: len(sim1), len(sim2) :])
    # This have shape [ n_samples x 1 ]
    logger.info(f"Shape before averaging: {sim.shape}")
    return sum(sim) / len(sim)
