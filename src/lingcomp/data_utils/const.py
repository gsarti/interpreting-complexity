""" Constants regarding datasets files and fields """

# GECO Eyetracking Dataset Constants

# Columns to be kept while importing Excel ET datasets
GECO_DATA_COLS = [
    "PP_NR",  # Participant identifier
    "PART",  # Part of novel identifier
    "TRIAL",  # Trial dentifier
    "WORD_ID",  # Format part-trial-word_id
    "WORD",
    "WORD_FIXATION_COUNT",
    "WORD_SKIP",
    "WORD_TOTAL_READING_TIME",
    "WORD_FIRST_FIXATION_DURATION",
    "WORD_GAZE_DURATION",
    "WORD_RUN_COUNT",
    "WORD_GO_PAST_TIME",
    "WORD_SELECTIVE_GO_PAST_TIME",
]

# Columns to be kept from the EnglishMaterial file
GECO_MATERIAL_COLS = ["WORD_ID", "PART_OF_SPEECH"]

# We want to replace missing durations with 0 to compute scores across participants.
# Missing POS tags are also filled with UNK.
GECO_NA_VALUES = {
    "WORD_TOTAL_READING_TIME": ".",
    "WORD_FIRST_FIXATION_DURATION": ".",
    "WORD_GAZE_DURATION": ".",
    "WORD_GO_PAST_TIME": ".",
    "WORD_SELECTIVE_GO_PAST_TIME": ".",
}

# Map from GECO-style POS tags to Universal POS used for other datasets (Dundee, ZuCo)
GECO_POS_MAP = {
    "Determiner": "DET",
    "Article": "DET",
    "Ex": "DET",
    "Adjective": "ADJ",
    "NA": "ADJ",  # The word 'False' in 'A False step', id 4-96-86
    "Verb": "VERB",
    "Noun": "NOUN",
    "Name": "NOUN",
    "Adverb": "ADV",
    "Not": "ADV",
    "Pronoun": "PRON",
    "Conjunction": "CONJ",
    "Preposition": "ADP",
    "To": "PRT",
    "Number": "NUM",
    ".": ".",
    "Interjection": "X",
    "Unclassified": "X",
    "Letter": "X",
    "UNK": "X",
}

# Dundee Corpus Eyetracking Constants

# Columns to be kept while importing Excel ET datasets
DUNDEE_DATA_COLS = [
    "Participant",
    "Itemno",
    "SentenceID",
    "ID",
    "WORD",
    "WLEN",
    "WNUM",
    "UniversalPOS",
    "nFix",
    "Fix_prob",
    "Mean_fix_dur",
    "First_fix_dur",
    "First_pass_dur",
    "Tot_fix_dur",
    "nRefix",
    "Re-read_prob",
    "Tot_regres_from_dur",
    "n-1_fix_prob",
    "n+1_fix_prob",
    "n-2_fix_prob",
    "n+2_fix_prob",
    "n-1_fix_dur",
    "n+1_fix_dur",
    "n-2_fix_dur",
    "n+2_fix_dur",
]

# Generic Eyetracking Dataset Constants

# Data types for the preprocessed output dataframe (per-participant scores)
OUT_TYPES_WORD = {
    "participant": str,
    "text_id": str,
    "sentence_id": str,
    "word_id": str,
    "word": str,
    "length": int,
    "pos": str,
    "fix_count": int,
    "fix_prob": int,
    "mean_fix_dur": float,
    "first_fix_dur": float,
    "first_pass_dur": float,
    "tot_fix_dur": float,
    "refix_count": int,
    "reread_prob": int,
    "tot_regr_from_dur": float,
    "n-2_fix_prob": int,
    "n-1_fix_prob": int,
    "n+1_fix_prob": int,
    "n+2_fix_prob": int,
    "n-2_fix_dur": float,
    "n-1_fix_dur": float,
    "n+1_fix_dur": float,
    "n+2_fix_dur": float,
}

# Output types when extracting data at sentence level
OUT_TYPES_SENTENCE = {
    "participant": str,
    "text_id": str,
    "sentence_id": int,
    "sentence": str,
    "token_count": int,
    "fix_count": float,
    "fix_prob": float,
    "mean_fix_dur": float,
    "first_fix_dur": float,
    "first_pass_dur": float,
    "tot_fix_dur": float,
    "refix_count": float,
    "reread_prob": float,
    "tot_regr_from_dur": float,
}

TEXT_COLUMNS = ["participant", "text_id", "word", "pos"]

FILLNA_COLUMNS = [
    "mean_fix_dur",
    "first_fix_dur",
    "first_pass_dur",
    "tot_fix_dur",
    "tot_regr_from_dur",
    "n-1_fix_dur",
    "n-2_fix_dur",
    "n+1_fix_dur",
    "n+2_fix_dur",
]
