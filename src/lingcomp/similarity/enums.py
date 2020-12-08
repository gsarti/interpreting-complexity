from enum import Enum


class SimilarityFunction(Enum):
    RSA = "RSA"
    PWCCA = "PWCCA"


class SimilarityStrategy(Enum):
    CLS_TOKEN = "cls_token"
    REDUCE_MEAN = "reduce_mean"
    PER_TOKEN = "per_token"


class SimilarityPlotMode(Enum):
    INTER = "inter"
    INTRA = "intra"
