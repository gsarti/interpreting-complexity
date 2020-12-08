import argparse
import logging
import os
import string

import pandas as pd
from sklearn.model_selection import train_test_split

from lingcomp.data_utils import DundeeProcessor, GECOProcessor, ZuCoProcessor
from lingcomp.script_utils import compute_agreement, reindex_sentence_df, save_tsv, train_test_split_sentences


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s  %(message)s", datefmt="%d-%m-%y %H:%M:%S", level=logging.INFO,
)

logger = logging.getLogger(__name__)

# From http://www.italianlp.it/resources/corpus-of-sentences-rated-with-human-complexity-judgments/
PC_DATA = "complexity/complexity_ds_en.csv"
# For eyetracking data, refer to et_processor.py
# OneStopEnglish official corpus CSV files
RA_FOLDER = "readability/raw_texts"

PC_FEATURES = "complexity/complexity_features.tsv"

ET_MAPPINGS = {
    "geco": GECOProcessor,
    "dundee": DundeeProcessor,
    "zuco1-nr": ZuCoProcessor,
    "zuco1-sr": ZuCoProcessor,
    "zuco2": ZuCoProcessor,
}


def preprocess_complexity_data(args):
    # Needed to avoid making the "null" word in text a NaN
    pc_file = os.path.join(args.data_dir, PC_DATA)
    pc = pd.read_csv(pc_file, na_values=["N/A"], keep_default_na=False)
    # Remove duplicates
    pc = pc[~pc.duplicated("SENTENCE")]
    pc_vals_start_idx = 2
    # Keep only annotations to compute agreement scores
    vals = pc.iloc[:, pc_vals_start_idx:]
    # Check if at least 10 participants agree on the complexity score
    agreement = [x >= args.complexity_min_agree for x in compute_agreement(vals, vals.mean(axis=1), vals.std(axis=1))]
    df = pd.DataFrame({"index": pc["ID"], "text": pc["SENTENCE"], "score": vals.mean(axis=1),})
    if args.do_features:
        # Load features
        pc_features_file = os.path.join(args.data_dir, PC_FEATURES)
        pc_features = pd.read_csv(pc_features_file, sep="\t")
        # Concatenate PC linguistic features
        df = pd.concat([df.reset_index(drop=True), pc_features.reset_index(drop=True)], axis=1)
    # Filter by agreement
    df = df[agreement]
    out = os.path.join(args.out_dir, "complexity_data.tsv")
    save_tsv(df, out)
    logger.info(f"Perceived complexity data were preprocessed and saved as" f" {out} with shape {df.shape}")
    return df


def preprocess_eyetracking_data(args):
    dfs = []
    for dataset in args.eyetracking_datasets:
        logging.info(f"Preprocessing eyetracking dataset: {dataset}")
        # do_features is used for GECO, version for ZuCO
        processor = ET_MAPPINGS[dataset](
            os.path.join(args.data_dir, "eyetracking"),
            do_features=args.do_features,
            version=dataset,
            fillna=args.fillna_strategy,
        )
        if args.eyetracking_mode == "word":
            df = processor.get_word_data(args.eyetracking_participant)
            # Perform word preprocessing to ensure wanted format
            # This is needed since we compute the loss only on the first subword after BERT tokenization
            df["word"] = [str(w).lstrip(string.punctuation) for w in df["word"]]
        elif args.eyetracking_mode == "sentence":
            df = processor.get_sentence_data(
                args.eyetracking_participant,
                min_len=args.eyetracking_min_sent_len,
                max_len=args.eyetracking_max_sent_len,
            )
            # Common format across sentence-level datasets used in FARM
            df.rename(columns={"sentence": "text"}, inplace=True)
        dfs.append(df)
    if len(dfs) > 1:
        df = pd.concat(dfs, ignore_index=True)
        df = reindex_sentence_df(df)
    m, p = args.eyetracking_mode, args.eyetracking_participant
    out = os.path.join(args.out_dir, f"eyetracking_data_{m}_{p}.tsv")
    save_tsv(df, out)
    logger.info(f"Eyetracking data were preprocessed and saved as" f" {out} with shape {df.shape}")
    return df


def preprocess_readability_data(args):
    ra_dir = os.path.join(args.data_dir, RA_FOLDER)
    idxs, texts, reading_levels = [], [], []
    for filename in os.listdir(ra_dir):
        if not filename.endswith(".txt"):
            continue
        name = filename.split("-")[0]
        with open(os.path.join(ra_dir, filename), "r") as f:
            label = f.readline()
            label = label.rstrip("\n")
            sentences = f.readlines()
            sentences = [s.rstrip("\n") for s in sentences]
            sentences = [s for s in sentences if s]
        idxs += [f"{name}-{i}" for i in range(1, len(sentences) + 1)]
        texts += sentences
        reading_levels += [label for i in range(len(sentences))]
    df = pd.DataFrame({"index": idxs, "text": texts, "reading_level": [l.strip() for l in reading_levels]})
    out = os.path.join(args.out_dir, "readability_data.tsv")
    save_tsv(df, out)
    logger.info(f"Readability assessment data were preprocessed and saved as" f" {out} with shape {df.shape}")
    return df


def do_train_test_split(args):
    logger.info("Performing train-test split for all datasets")
    task_dic = {"complexity": args.complexity, "eyetracking": args.eyetracking, "readability": args.readability}
    for task, do_task in task_dic.items():
        folder = f"{args.data_dir}/{task}/train_test"
        if not os.path.exists(folder) and do_task:
            os.makedirs(folder)
            if task == "complexity":
                train, test = train_test_split(args.pc, test_size=args.test_size, random_state=args.seed)
            elif task == "eyetracking":
                if args.eyetracking_mode == "word":
                    train, test = train_test_split_sentences(
                        args.et, test_frac=args.test_size, sentenceid_col="sentence_id"
                    )
                elif args.eyetracking_mode == "sentence":
                    train, test = train_test_split(args.et, test_size=args.test_size, random_state=args.seed)
            elif task == "readability":
                train, test = train_test_split(
                    args.ra, test_size=args.test_size, random_state=args.seed, stratify=args.ra[["reading_level"]]
                )
            save_tsv(train, f"{folder}/train.tsv")
            save_tsv(test, f"{folder}/test.tsv")
            logger.info(f"Train-test data saved in {folder}")
        else:
            if do_task:
                logger.info("Train-test data already exist in path, not overriding them.")


def preprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Shorthand to preprocess all datasets")
    parser.add_argument(
        "--all_et_datasets", action="store_true", help="Shorthand to use all available eye-tracking datasets."
    )
    parser.add_argument("--complexity", action="store_true", help="Preprocess the perceived complexity corpus.")
    parser.add_argument("--eyetracking", action="store_true", help="Preprocess the eye-tracking corpora.")
    parser.add_argument("--readability", action="store_true", help="Preprocess the readability corpus")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Folder containing the three subfolders for each dataset."
    )
    parser.add_argument("--out_dir", type=str, default="data/preprocessed")
    parser.add_argument("--do_train_test_split", action="store_true")
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--complexity_min_agree", type=int, default=10, help="Minimum IAA for keeping perceived complexity sentences.",
    )
    parser.add_argument(
        "--eyetracking_mode",
        type=str,
        default="word",
        choices=["word", "sentence"],
        help="One among `word` or `sentence`. Specifies metrics aggregation level",
    )
    parser.add_argument(
        "--eyetracking_participant",
        type=str,
        default="avg",
        help="Participant selected for eyetracking scores. Default is avg of all participants.",
    )
    parser.add_argument(
        "--eyetracking_datasets",
        nargs="+",
        choices=list(ET_MAPPINGS.keys()),
        default=["geco"],
        help="Datasets to be used for eye-tracking. Choices: %(choice)s, default: %(default)s"
        "If multiple are specified, they are merged into a single dataset.",
    )
    parser.add_argument("--eyetracking_min_sent_len", type=int, default=5)
    parser.add_argument("--eyetracking_max_sent_len", type=int, default=45)
    parser.add_argument("--do_features", action="store_true")
    parser.add_argument(
        "--fillna_strategy",
        default="zero",
        type=str,
        help="Specifies the NaN filling strategy for eyetracking processors.",
        choices=["none", "zero", "min_participant", "mean_participant", "max_participant"],
    )
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if args.all:
        args.complexity, args.eyetracking, args.readability = True, True, True
    if args.all_et_datasets:
        args.eyetracking_datasets = list(ET_MAPPINGS.keys())
    if args.complexity:
        args.pc = preprocess_complexity_data(args)
    if args.eyetracking:
        args.et = preprocess_eyetracking_data(args)
    if args.readability:
        args.ra = preprocess_readability_data(args)
    if args.do_train_test_split:
        do_train_test_split(args)


if __name__ == "__main__":
    preprocess()
