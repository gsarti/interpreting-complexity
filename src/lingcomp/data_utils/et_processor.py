import csv
import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from lingcomp.data_utils.const import (
    DUNDEE_DATA_COLS,
    FILLNA_COLUMNS,
    GECO_DATA_COLS,
    GECO_MATERIAL_COLS,
    GECO_NA_VALUES,
    GECO_POS_MAP,
    OUT_TYPES_SENTENCE,
    OUT_TYPES_WORD,
    TEXT_COLUMNS,
)
from lingcomp.data_utils.matlab_utils import read_zuco1_mat, read_zuco2_mat
from lingcomp.script_utils import apply_parallel, read_tsv, save_tsv


logger = logging.getLogger(__name__)


class EyetrackingProcessor:
    """ Abstraction for a reader that converts eye-tracking data in a preprocessed format """

    def __init__(
        self,
        data_dir,
        data_filename,
        out_filename,
        ref_participant,
        text_columns=TEXT_COLUMNS,
        out_types_word=OUT_TYPES_WORD,
        out_types_sentence=OUT_TYPES_SENTENCE,
        nan_cols=FILLNA_COLUMNS,
        fillna="zero",
        **kwargs,
    ):
        """
        data_dir: Directory where eye-tracking dataset and materials are contained.
        data_filename: Name of main file in the data_dir containing eye-tracking measurements.
        out_filename: File where the preprocessed output will be saved.
        text_columns: Names of columns to be treated as text during aggregation
        ref_participant: The name of the reference participant having annotated all examples,
            used for grouping and averaging scores.
        out_types_word: Dictionary of data types of word-level preprocessed data,
            with entries structured as column name : data type.
        out_types_sentence: Dictionary of data types of sentence-level preprocessed data,
            with entries structured as column name : data type.
        nan_cols: List of column names for columns that can possibly include NaN values.
        fillna: Specifies the fill-NaN strategy enacted during aggregation in get_word_data and get_sentence_data. Default: zero.
            Choose one among:
                - none: leaves NaNs as-is.
                - zero: fills NaNs with 0 => missing duration will count as 0 during averaging.
                - (min|mean|max)_participant: fills NaNs with the min|mean|max value for that token across participants.
            To be added in the future:
                - (min|mean|max)_type: fills NaNs with the min|mean|max value for that token in the whole dataset.
        """
        self.data_dir = data_dir
        self.data_path = os.path.join(data_dir, data_filename)
        self.out_preprocessed = os.path.join(data_dir, out_filename)
        self.out_cleaned = os.path.join(data_dir, f"fillna_{fillna}_{out_filename}")
        self.text_columns = text_columns
        self.ref_participant = ref_participant
        self.out_types_word = out_types_word
        self.out_types_sentence = out_types_sentence
        self.preprocessed_data = None
        logger.info(f"Unused arguments: {kwargs}")
        if not os.path.exists(self.out_preprocessed):
            logger.info("Preprocessing dataset, this may take some time...")
            self.create_preprocessed_dataset()
            logger.info("Done preprocessing.")
        logger.info(f"Loading preprocessed data from {self.out_preprocessed}")
        self.preprocessed_data = read_tsv(self.out_preprocessed)
        if not os.path.exists(self.out_cleaned):
            # We fill missing value following the specified strategy
            logger.info(f"Filling NaN values using strategy: {fillna}")
            self.fill_nan_values(fillna, nan_cols)
            logger.info("Done filling NaNs")
        logger.info(f"Loading cleaned data from {self.out_cleaned}")
        self.cleaned_data = read_tsv(self.out_cleaned)

    def fill_nan_values(self, strategy, columns):
        """ Fills NaNs in self.preprocessed_data based on strategy """
        if strategy == "none":
            df = self.preprocessed_data
        elif strategy == "zero":
            df = self.preprocessed_data.fillna({c: 0 for c in columns})
        elif strategy == "min_participant":
            # Slow for huge dataframes, can probably be improved
            # If a value is null for all participants, it is filled with 0s
            df = apply_parallel(self.preprocessed_data.groupby("word_id", sort=False), fillna_min, columns)
        elif strategy == "mean_participant":
            df = apply_parallel(self.preprocessed_data.groupby("word_id", sort=False), fillna_mean, columns)
        elif strategy == "max_participant":
            df = apply_parallel(self.preprocessed_data.groupby("word_id", sort=False), fillna_max, columns)
        else:
            raise AttributeError(f"Strategy {strategy} not supported yet.")
        df = df.sort_index()
        assert all(
            [x == y for x, y in zip(self.preprocessed_data["word_id"], df["word_id"])]
        ), "Word id order mismatch, fillna misbehaved."
        save_tsv(df, self.out_cleaned)

    def create_preprocessed_dataset(self):
        """ Must be implemented by subclasses """
        raise NotImplementedError

    def get_word_data(self, participant="avg"):
        # By default we average scores across participants
        if participant == "avg":
            # Compute average for each word across participants
            scores = self.cleaned_data.groupby("word_id", sort=False, as_index=False).mean()
            # Textual fields not considered in avg
            txt = self.cleaned_data[self.text_columns]
            # ref_participant has all words annotated, we replace it with 'avg' in the data field
            txt = txt[txt.participant == self.ref_participant]
            txt.participant = "avg"
            # Join numeric and textual fields, remove possible duplicate columns
            avg_data = pd.concat([txt.reset_index(), scores.reset_index()], axis=1, sort=False)
            avg_data = avg_data.loc[:, ~avg_data.columns.duplicated()]
            # Order columns in preprocessed shape
            avg_data = avg_data[list(self.out_types_word.keys())]
            return avg_data
        else:  # Data cleaned with fillna strategy are used for single participants, too
            subset = self.cleaned_data[self.cleaned_data["participant"] == participant]
            if len(subset) == 0:
                raise AttributeError(
                    f"Participant was not found in the dataset. Please choose one"
                    f" among: avg, {', '.join(list(set(self.preprocessed_data['participant'])))}"
                )
            return subset

    def get_sentence_data(self, participant="avg", min_len=-1, max_len=100000):
        # By default we average scores across participants
        data = self.get_word_data(participant)
        # Compute average for each word across participants
        group_sent = data.groupby("sentence_id", sort=False, as_index=False)
        scores = group_sent.sum()
        # Textual fields not considered in avg
        scores["sentence"] = list(group_sent["word"].apply(lambda v: " ".join([str(y) for y in v])))
        scores["participant"] = [participant for i in range(len(scores))]
        scores["text_id"] = list(group_sent["text_id"].first()["text_id"])
        # Add token counts per sentence using whitespace tokenization
        scores["token_count"] = list(group_sent["word"].apply(len))
        scores = scores.astype(self.out_types_sentence)
        scores = scores[list(self.out_types_sentence.keys())]
        # Filter based on whitespace token count (not the same as n_token from features!)
        scores = scores[(scores["token_count"] <= max_len) & (scores["token_count"] >= min_len)]
        return scores


class GECOProcessor(EyetrackingProcessor):
    """ Reader that converts the GECO dataset in a preprocessed format. """

    def __init__(self, data_dir, do_features=True, **kwargs):
        # Named after original files from http://expsy.ugent.be/downloads/geco/
        self.materials_path = os.path.join(data_dir, "EnglishMaterial.xlsx")
        # Those are generated files provided in the repository
        self.sentence_ids_path = os.path.join(data_dir, "sentence_ids.tsv")
        self.features_path = os.path.join(data_dir, "geco_features.tsv")
        self.do_features = do_features
        super(GECOProcessor, self).__init__(
            data_dir,
            data_filename="MonolingualReadingData.xlsx",
            out_filename="preprocessed_geco.tsv",
            ref_participant="pp21",
            **kwargs,
        )

    def create_preprocessed_dataset(self):
        data = pd.read_excel(
            self.data_path, usecols=GECO_DATA_COLS, sheet_name="DATA", na_values=GECO_NA_VALUES, keep_default_na=False,
        )
        extra = pd.read_excel(
            self.materials_path, sheet_name="ALL", na_values=["N/A"], keep_default_na=False, usecols=GECO_MATERIAL_COLS
        )
        sent_ids = read_tsv(self.sentence_ids_path)
        logger.info("Preprocessing values for the dataset...")
        df = pd.merge(data, extra, how="left", on="WORD_ID")
        df = pd.merge(df, sent_ids, how="left", on="WORD_ID")
        # Clean up words since we need to rely on whitespaces for aligning
        # sentences with tokens.
        df["WORD"] = [str(w).replace(" ", "") for w in df["WORD"]]
        # Create new fields for the dataset
        text_id = [f"{x}-{y}" for x, y in zip(df["PART"], df["TRIAL"])]
        length = [len(str(x)) for x in df["WORD"]]
        # Handle the case where we don't fill NaN values
        mean_fix_dur = []
        for x, y in zip(df["WORD_TOTAL_READING_TIME"], df["WORD_FIXATION_COUNT"]):
            if pd.isna(x):
                mean_fix_dur.append(np.nan)
            elif y == 0:
                mean_fix_dur.append(0)
            else:
                mean_fix_dur.append(x / y)
        refix_count = [max(x - 1, 0) for x in df["WORD_RUN_COUNT"]]
        reread_prob = [x > 1 for x in df["WORD_FIXATION_COUNT"]]
        # Handle the case where we don't fill NaN values
        tot_regr_from_dur = []
        for x, y in zip(df["WORD_GO_PAST_TIME"], df["WORD_SELECTIVE_GO_PAST_TIME"]):
            if pd.isna(x) or pd.isna(y):
                tot_regr_from_dur.append(np.nan)
            else:
                tot_regr_from_dur.append(max(x - y, 0))
        # 2050 tokens per participant do not have POS info.
        # We use a special UNK token for missing pos tags.
        pos = [GECO_POS_MAP[x] if not pd.isnull(x) else GECO_POS_MAP["UNK"] for x in df["PART_OF_SPEECH"]]
        fix_prob = [1 - x for x in df["WORD_SKIP"]]
        # Format taken from Hollenstein et al. 2019 "NER at First Sight"
        out = pd.DataFrame(
            {
                # Identifiers
                "participant": df["PP_NR"],
                "text_id": text_id,  # PART-TRIAL for GECO
                "sentence_id": df["SENTENCE_ID"],  # Absolute sentence position for GECO
                # AOI-level measures
                "word_id": df["WORD_ID"],
                "word": df["WORD"],
                "length": length,
                "pos": pos,
                # Basic measures
                "fix_count": df["WORD_FIXATION_COUNT"],
                "fix_prob": fix_prob,
                "mean_fix_dur": mean_fix_dur,
                # Early measures
                "first_fix_dur": df["WORD_FIRST_FIXATION_DURATION"],
                "first_pass_dur": df["WORD_GAZE_DURATION"],
                # Late measures
                "tot_fix_dur": df["WORD_TOTAL_READING_TIME"],
                "refix_count": refix_count,
                "reread_prob": reread_prob,
                # Context measures
                "tot_regr_from_dur": tot_regr_from_dur,
                "n-2_fix_prob": ([0, 0] + fix_prob)[: len(df)],
                "n-1_fix_prob": ([0] + fix_prob)[: len(df)],
                "n+1_fix_prob": (fix_prob + [0])[1:],
                "n+2_fix_prob": (fix_prob + [0, 0])[2:],
                "n-2_fix_dur": ([0, 0] + list(df["WORD_TOTAL_READING_TIME"]))[: len(df)],
                "n-1_fix_dur": ([0] + list(df["WORD_TOTAL_READING_TIME"]))[: len(df)],
                "n+1_fix_dur": (list(df["WORD_TOTAL_READING_TIME"]) + [0])[1:],
                "n+2_fix_dur": (list(df["WORD_TOTAL_READING_TIME"]) + [0, 0])[2:],
            }
        )
        # Convert to correct data types
        out = out.astype(self.out_types_word)
        # Caching preprocessed dataset for next Processor calls
        save_tsv(out, self.out_preprocessed)
        logger.info(f"GECO data were preprocessed and saved as" f" {self.out_preprocessed} with shape {out.shape}")
        self.preprocessed_data = out

    def get_sentence_data(self, participant="avg", min_len=5, max_len=45):
        scores = super(GECOProcessor, self).get_sentence_data(participant)
        if self.do_features:
            # Load ET features
            et_features = pd.read_csv(self.features_path, sep="\t")
            scores = pd.concat([scores.reset_index(drop=True), et_features.reset_index(drop=True)], axis=1)
        # Filter based on whitespace token count (not the same as n_token from features!)
        scores = scores[(scores["token_count"] <= max_len) & (scores["token_count"] >= min_len)]
        # Order columns in preprocessed shape
        return scores


class DundeeProcessor(EyetrackingProcessor):
    """Reader that converts the Dundee dataset in a preprocessed format.
    We already start from a preprocessed version provided by Nora Hollenstein, available upon request.
    """

    def __init__(self, data_dir, **kwargs):
        super(DundeeProcessor, self).__init__(
            data_dir,
            data_filename="EN_dundee.tsv",
            out_filename="preprocessed_dundee.tsv",
            ref_participant="sa",
            **kwargs,
        )

    def create_preprocessed_dataset(self):
        df = pd.read_csv(
            self.data_path,
            usecols=DUNDEE_DATA_COLS,
            sep="\t",
            quoting=csv.QUOTE_NONE,
            engine="python",
            na_values=[""],
            keep_default_na=False,
        )
        # Clean up words since we need to rely on whitespaces for aligning
        # sentences with tokens.
        df["WORD"] = [str(w).replace(" ", "") for w in df["WORD"]]
        logger.info("Preprocessing values for the dataset...")
        keep_idx = []
        curr_sent_id, curr_wnum = 1, 0
        curr_val = df.loc[0, "SentenceID"]
        curr_pp = df.loc[0, "Participant"]
        sent_ids, word_ids = [], []
        for _, r in tqdm(df.iterrows()):
            # Tokens are split from punctuation for POS tagging, we need to reassemble regions.
            # We use WNUM to check if the token belongs to the same region.
            if r["WNUM"] == curr_wnum:
                keep_idx.append(False)
                continue
            keep_idx.append(True)
            curr_wnum = r["WNUM"]
            # Advance sentence id
            if r["SentenceID"] != curr_val:
                curr_sent_id += 1
                curr_val = r["SentenceID"]
            # Data are ordered, so we can reset sentence indexes when switching participants
            if r["Participant"] != curr_pp:
                curr_sent_id = 1
                curr_pp = r["Participant"]
            sent_ids.append(curr_sent_id)
            word_ids.append(f'{int(r["Itemno"])}-{int(r["SentenceID"])}-{int(r["ID"])}')
        # Filter out duplicates
        df = df[keep_idx]
        out = pd.DataFrame(
            {
                # Identifiers
                "participant": df["Participant"],
                "text_id": df["Itemno"],
                "sentence_id": sent_ids,
                # AOI-level measures
                "word_id": word_ids,
                "word": df["WORD"],
                "length": df["WLEN"],
                "pos": df["UniversalPOS"],
                # Basic measures
                "fix_count": df["nFix"],
                "fix_prob": df["Fix_prob"],
                "mean_fix_dur": df["Mean_fix_dur"],
                # Early measures
                "first_fix_dur": df["First_fix_dur"],
                "first_pass_dur": df["First_pass_dur"],
                # Late measures
                "tot_fix_dur": df["Tot_fix_dur"],
                "refix_count": df["nRefix"],
                "reread_prob": df["Re-read_prob"],
                # Context measures
                "tot_regr_from_dur": df["Tot_regres_from_dur"],
                "n-2_fix_prob": df["n-2_fix_prob"],
                "n-1_fix_prob": df["n-1_fix_prob"],
                "n+1_fix_prob": df["n+1_fix_prob"],
                "n+2_fix_prob": df["n+2_fix_prob"],
                "n-2_fix_dur": df["n-2_fix_dur"],
                "n-1_fix_dur": df["n-1_fix_dur"],
                "n+1_fix_dur": df["n+1_fix_dur"],
                "n+2_fix_dur": df["n+2_fix_dur"],
            }
        )
        # Convert to correct data types
        out = out.astype(self.out_types_word)
        # Caching preprocessed dataset for next Processor calls
        save_tsv(out, self.out_preprocessed)
        logger.info(f"Dundee data were preprocessed and saved as" f" {self.out_preprocessed} with shape {out.shape}")
        self.preprocessed_data = out


class ZuCoProcessor(EyetrackingProcessor):
    """ Reader that converts ZuCo v1/v2 .mat files in a preprocessed format. """

    def __init__(self, data_dir, version="zuco1-nr", **kwargs):
        self.version = version
        self.mat_files_path = os.path.join(data_dir, version)
        out_filename = f"preprocessed_{version}.tsv"
        ref_participant = "YDR" if version == "zuco2" else "ZKB"
        super(ZuCoProcessor, self).__init__(
            data_dir, data_filename="", out_filename=out_filename, ref_participant=ref_participant, **kwargs
        )

    def create_preprocessed_dataset(self):
        if self.version in ["zuco1-nr", "zuco1-sr"]:
            df = read_zuco1_mat(self.mat_files_path)
        elif self.version == "zuco2":
            df = read_zuco2_mat(self.mat_files_path)
        else:
            raise AttributeError("Selected version of ZuCo does not exist.")
        # Clean up words since we need to rely on whitespaces for aligning
        # sentences with tokens.
        logger.info("Preprocessing values for the dataset...")
        df["content"] = [str(w).replace(" ", "") for w in df["content"]]
        word_skip = [int(v) for v in list(df["FXC"].isna())]
        # If FXC is NaN, it corresponds to 0 fixations
        df["FXC"] = df["FXC"].fillna(0)
        # Create new fields for the dataset
        word_id = [f"{x}-{y}-{z}" for x, y, z in zip(df["task_id"], df["sent_idx"], df["word_idx"].astype("int32"))]
        length = [len(str(x)) for x in df["content"]]
        mean_fix_dur = []
        for x, y in zip(df["TRT"], df["FXC"]):
            if pd.isna(x) or pd.isna(y):
                mean_fix_dur.append(np.nan)
            elif y == 0:
                mean_fix_dur.append(0)
            else:
                mean_fix_dur.append(x / y)
        refix_count = [max(x - 1, 0) if pd.notna(x) else np.nan for x in df["FXC"]]
        reread_prob = [x > 1 if pd.notna(x) else np.nan for x in df["FXC"]]
        # Since here we do not have the selective go past time as for GECO,
        # we approximate it using go-past time minus gaze duration.
        # Note that this approximation is a lower bound in case of multiple regressions.
        tot_regr_from_dur = []
        for x, y in zip(df["GPT"], df["GD"]):
            if pd.isna(x) or pd.isna(y):
                tot_regr_from_dur.append(np.nan)
            else:
                tot_regr_from_dur.append(max(x - y, 0))
        # We do not have POS info for ZuCo corpora
        pos = ["UNK" for x in range(len(df))]
        fix_prob = [1 - x for x in word_skip]
        # Format taken from Hollenstein et al. 2019 "NER at First Sight"
        out = pd.DataFrame(
            {
                # Identifiers
                "participant": df["participant"],
                "text_id": df["task_id"],  # Name of the recorded reading portion
                "sentence_id": df["sent_idx"],  # Absolute sentence position in reading portion
                # AOI-level measures
                "word_id": word_id,
                "word": df["content"],
                "length": length,
                "pos": pos,
                # Basic measures
                "fix_count": df["FXC"],
                "fix_prob": fix_prob,
                "mean_fix_dur": mean_fix_dur,
                # Early measures
                "first_fix_dur": df["FFD"],
                "first_pass_dur": df["GD"],
                # Late measures
                "tot_fix_dur": df["TRT"],
                "refix_count": refix_count,
                "reread_prob": reread_prob,
                # Context measures
                "tot_regr_from_dur": tot_regr_from_dur,
                "n-2_fix_prob": ([0, 0] + fix_prob)[: len(df)],
                "n-1_fix_prob": ([0] + fix_prob)[: len(df)],
                "n+1_fix_prob": (fix_prob + [0])[1:],
                "n+2_fix_prob": (fix_prob + [0, 0])[2:],
                "n-2_fix_dur": ([0, 0] + list(df["TRT"]))[: len(df)],
                "n-1_fix_dur": ([0] + list(df["TRT"]))[: len(df)],
                "n+1_fix_dur": (list(df["TRT"]) + [0])[1:],
                "n+2_fix_dur": (list(df["TRT"]) + [0, 0])[2:],
            }
        )
        # Convert to correct data types
        out = out.astype(self.out_types_word)
        # Caching preprocessed dataset for next Processor calls
        save_tsv(out, self.out_preprocessed)
        logger.info(
            f"{self.version} data were preprocessed and saved as" f" {self.out_preprocessed} with shape {out.shape}"
        )
        self.preprocessed_data = out


def fillna_min(columns, df):
    return df.fillna({k: v for k, v in zip(columns, df[columns].min().fillna(0))})


def fillna_mean(columns, df):
    return df.fillna({k: v for k, v in zip(columns, df[columns].mean().fillna(0))})


def fillna_max(columns, df):
    return df.fillna({k: v for k, v in zip(columns, df[columns].max().fillna(0))})
