"""
Get surprisal estimates for a transformers model.
Adapted from https://github.com/cpllab/lm-zoo/blob/master/models/transformers-base/get_surprisals.py
Now using FARM tokenizers and language models to allow for whitespace-tokenized aggregation
"""

import argparse
import logging

import numpy as np
import pandas as pd
import torch
from farm.modeling.language_model import LanguageModel
from farm.modeling.tokenization import tokenize_with_metadata
from tqdm import tqdm
from transformers.modeling_auto import AutoModelWithLMHead
from transformers.tokenization_auto import AutoTokenizer

from lingcomp.farm.tokenization import CustomTokenizer
from lingcomp.script_utils import get_sentences_from_json, read_tsv, save_tsv, set_seed


logger = logging.getLogger(__name__)


def _get_predictions_inner(sentence, tokenizer, model, device):
    meta = tokenize_with_metadata(sentence, tokenizer)
    sent_tokens, offsets, start_of_words = meta["tokens"], meta["offsets"], meta["start_of_word"]
    indexed_tokens = tokenizer.convert_tokens_to_ids(sent_tokens)
    # create 1 * T input token tensor
    tokens_tensor = torch.tensor(indexed_tokens).unsqueeze(0)
    tokens_tensor = tokens_tensor.to(device)
    with torch.no_grad():
        log_probs = model(tokens_tensor)[0].log_softmax(dim=2).squeeze()
    return list(zip(sent_tokens, indexed_tokens, (None,) + log_probs.unbind(), offsets, start_of_words))


def get_surprisal_scores(sentence, tokenizer, model, device):
    predictions = _get_predictions_inner(sentence, tokenizer, model, device)
    surprisals = []
    for token, token_idx, preds, offset, sow in predictions:
        if preds is None:
            surprisal = 0.0
        else:
            surprisal = -preds[token_idx].item() / np.log(2)
        surprisals.append((token, token_idx, surprisal, offset, sow))
    return surprisals


def aggregate_word_level(sentence, surprisals):
    word_spans, word_surps = [], []
    span, surp = None, 0
    for token, token_idx, surprisal, offset, start_of_word in surprisals:
        if start_of_word:
            # previous word has ended unless it's the very first word
            if span is not None:
                word_spans.append(span)
                word_surps.append(surp)
            span = {"start": offset, "end": offset + len(token)}
            surp = surprisal
        else:
            # expand the span to include the subword-token
            span["end"] = offset + len(token.replace("##", ""))
            surp += surprisal
    word_spans.append(span)
    word_surps.append(surp)
    words = [sentence[span["start"] : span["end"]].rstrip() for span in word_spans]
    assert len(words) == len(word_surps), "Word-surprisal number mismatch"
    return words, word_surps, word_spans


def get_surprisals(args):
    set_seed(args.seed, cuda=args.cuda)
    logger.info("Importing tokenizer and pre-trained model")
    tok_class = None if not args.model_class_name else f"{args.model_class_name}Tokenizer"
    ref = args.reference_hf_model if args.reference_hf_model is not None else args.model_name_or_path
    model = AutoModelWithLMHead.from_pretrained(ref)
    # Loading a local model, we need to replace the AutoModel with the local model
    if args.reference_hf_model is not None:
        farm_lm = LanguageModel.load(args.model_name_or_path, language_model_class=args.model_class_name)
        # Set the underlying model to the custom loaded model
        # The LM head used for surprisal is the original pretrained head
        logger.info(f"Setting model.{model.base_model_prefix} attribute with model: {args.model_name_or_path}")
        setattr(model, model.base_model_prefix, farm_lm.model)
        tokenizer = CustomTokenizer.load(
            pretrained_model_name_or_path=args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            tokenizer_class=tok_class,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(ref)
    device = torch.device("cuda" if args.cuda else "cpu")
    model.to(device)
    model.eval()
    logger.info(f"Reading sentences from {args.inputf}")
    if args.inputf.endswith(".tsv"):  # lingcomp tsv format
        df = read_tsv(args.inputf)
        sentences = list(df["text"])
    elif args.inputf.endswith(".json"):  # syntaxgym test suite format
        sentences = get_sentences_from_json(args.inputf)
    elif args.inputf.endswith(".txt"):  # one sentencen per line
        sentences = open(args.inputf, "r").read().split("\n")
    else:
        raise AttributeError("Only .tsv, .json and .txt input files are supported.")
    dict_list = []
    for i, sentence in tqdm(enumerate(sentences)):
        surprisals = get_surprisal_scores(sentence, tokenizer, model, device)
        if args.mode in ["token", "sentence"]:
            for token, token_idx, surprisal, _, _ in surprisals:
                dict_list.append({"sentence_id": i + 1, "token_id": token_idx, "token": token, "surprisal": surprisal})
        elif args.mode == "word":
            words, word_surps, word_spans = aggregate_word_level(sentence, surprisals)
            for j, word in enumerate(words):
                dict_list.append(
                    {
                        "start": word_spans[j]["start"],
                        "end": word_spans[j]["end"],
                        "context": word,
                        "surprisal": word_surps[j],
                        "sentence_id": i + 1,
                        "token_id": j + 1,
                    }
                )
    out = pd.DataFrame(dict_list)
    if args.mode == "sentence":
        surprisals = list(out.groupby("sentence_id", sort=False).sum()["surprisal"])
        assert len(surprisals) == len(sentences), "Sentence-surprisal number mismatch"
        dict_list = []
        for k, sent in enumerate(sentences):
            dict_list.append({"sentence_id": k + 1, "sentence": sent, "surprisal": surprisals[k]})
        out = pd.DataFrame(dict_list)
    logger.info(f"Surprisal values at {args.mode}-level were saved to {args.outputf}")
    save_tsv(out, args.outputf)


def main():
    parser = argparse.ArgumentParser(description="Get surprisal estimates at different levels of granularity")
    parser.add_argument(
        "--inputf",
        type=str,
        required=True,
        help="Input file containing sentences."
        "- If .txt one sentence per line is assumed."
        "- If .tsv, assumes that can be read with read_tsv method and that sentences are located in the 'text' field."
        "- If .json, assumes it is a test suite in SyntaxGym format and loads it appropriately.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to model directory containing files in FARM format, or name from one of the supported models in the"
        "HuggingFace hub.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--model_class_name",
        default=None,
        type=str,
        help="Name of the model class in FARM format, used for disambiguating model matching when required. Needed only in case of warnings."
        "Example: Loading a local BERT from a repository which doesn't include 'bert' in the name, this must be set to BERTModel.",
    )
    parser.add_argument(
        "--reference_hf_model",
        default=None,
        type=str,
        help="When loading a custom fine-tuned model, we must load the LM prediction head from the respective HuggingFace model."
        "Example: Loading a local BERT fine-tuned on NER with FARM, this param must be set to e.g. bert-base-cased, depending on the original model.",
    )
    parser.add_argument("--cuda", default=False, action="store_true", help="Toggle cuda to run on GPU")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--outputf", "-o", type=str, default="logs/out_surprisal.tsv", help="output file for generated text"
    )
    parser.add_argument("--mode", choices=["token", "word", "sentence"], default="token")
    get_surprisals(parser.parse_args())


if __name__ == "__main__":
    main()
