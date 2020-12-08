import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from lingcomp.args.default_args import DefaultArguments
from lingcomp.similarity.enums import SimilarityFunction, SimilarityPlotMode, SimilarityStrategy


logger = logging.getLogger(__name__)


@dataclass
class SimilarityArguments(DefaultArguments):
    similarity_function: SimilarityFunction = field(
        default="rsa", metadata={"help": "Similarity function that should be used to compare embeddings."}
    )
    model_a: Optional[str] = field(
        default=None, metadata={"help": "Local directory or HF hosting name of the first model to load."}
    )
    model_b: Optional[str] = field(
        default=None, metadata={"help": "Local directory or HF hosting name of the second model to load."}
    )
    data_path: Optional[str] = field(
        default="data/eval/sst.tsv", metadata={"help": "Path to the file containing test input."}
    )
    data_name: Optional[str] = field(default=None, metadata={"help": "Data identifier, for save-naming purposes"})
    save_dir: Optional[str] = field(
        default="logs/similarity_scores", metadata={"help": "The directory where results must be saved."}
    )
    batch_size: Optional[int] = field(default=512, metadata={"help": "Batch size per GPU/CPU for training."})
    strategy: SimilarityStrategy = field(
        default="cls_token", metadata={"help": "Embedding extraction strategy used for model comparison."}
    )
    start_layer: Optional[int] = field(
        default=-1,
        metadata={
            "help": "Starting layer for which the similarity should be computed. Default -1 (last layer). Accepted values are in the range of layers (-1 to -12 for base models)."
        },
    )
    end_layer: Optional[int] = field(
        default=-1,
        metadata={
            "help": "Ending layer for which the similarity should be computed. Default -1 (last layer). Accepted values are in the range of layers (-1 to -12 for base models)."
        },
    )
    cache_acts: bool = field(default=False, metadata={"help": "Save models activations inside pickled files."})
    save_results: bool = field(
        default=False, metadata={"help": "Set this flag if you want to save results to a tsv file."}
    )
    # Plot mode
    plot: bool = field(default=False, metadata={"help": "Use plot mode."})
    plot_dir: Optional[str] = field(
        default="img/similarity_plots", metadata={"help": "Directory where generated plots should be saved."}
    )
    plot_mode: SimilarityPlotMode = field(
        default="inter",
        metadata={
            "help": "Plot mode, depending on similarity approach. PWCCA: `inter` to show the progression of PWCCA distances across layers, `intra` to show distances inside a single layer. RSA: `inter` to show distances for each layer across models, `intra` to show distances across layers of a single model."
        },
    )
    use_aliases: bool = field(
        default=False, metadata={"help": "If set, asks user for model aliases to be used inside plots."}
    )
    # PWCCA-specific args
    avg_pseudodist: bool = field(
        default=False,
        metadata={
            "help": "Since PWCCA isn't symmetrical, d(A,B) != d(B,A). By default, the order specified in the parameters is used. Setting this parameter computes the average of scores for both orders as the final score."
        },
    )

    def __post_init__(self):
        if not self.plot and (self.model_a is None or self.model_b is None):
            raise AttributeError("Models should be specified in compute mode, use --plot for plot mode.")
        if self.data_name is None:
            self.data_name = self.data_path.split("/")[-1].split(".")[0]
        if self.plot and not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        if not self.plot and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
