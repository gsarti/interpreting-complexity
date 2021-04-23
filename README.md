# Interpreting Models of Linguistic Complexity

This repository contains data and code implementations for reproducing all the experiments for:

**Interpreting Neural Language Models for Linguistic Complexity Assessment**, [Gabriele Sarti](https://gsarti.com), *Data Science and Scientific Computing MSc Thesis, University of Trieste, 2020* [[Gitbook]](https://gsarti.com/thesis/introduction.html) [[Slides (Long)](https://drive.google.com/file/d/1mb_Wlzrvog5-eds6hcSrm7gHSj9PO6qw/view?usp=sharing)] [[Slides (Short)](https://drive.google.com/file/d/1j2zCavx4EzomRIoTwmtvvmGbWizKmHEA/view?usp=sharing)]

**UmBERTo-MTSA @ AcCompl-It: Improving Complexity and Acceptability Prediction with Multi-task Learning on Self-Supervised Annotations**, [Gabriele Sarti](https://gsarti.com), *Proceedings of Seventh Evaluation Campaign of Natural Language Processing and Speech Tools for Italian*, [[ArXiv](https://arxiv.org/abs/2011.05197)] [CEUR](http://ceur-ws.org/Vol-2765/paper163.pdf) [Video](https://vimeo.com/487817662)

**That Looks Hard: Characterizing Linguistic Complexity in Humans and Language Models**, [Gabriele Sarti](https://gsarti.com) and [Dominique Brunato](https://scholar.google.com/citations?user=JJV9ay4AAAAJ&hl=it) and [Felice Dell'Orletta](https://scholar.google.com/citations?user=uhInFTQAAAAJ&hl=it), *Proceeding of the Workshop on Cognitive Modeling and Computational Linguistics at NAACL 2021* [ACL Anthology]

If you find these resource useful for your research, please consider citing one or more following works:

```bibtex
@mastersthesis{sarti-2020-interpreting,
    author = {Sarti, Gabriele},
    institution = {University of Trieste},
    school = {University of Trieste},
    title = {Interpreting Neural Language Models for Linguistic Complexity Assessment},
    year = 2020
}

@inproceedings{sarti-2020-umbertomtsa,
    author = {Sarti, Gabriele},
    title = {{UmBERTo-MTSA @ AcCompl-It}: Improving Complexity and Acceptability Prediction with Multi-task Learning on Self-Supervised Annotations},
    booktitle = {Proceedings of Seventh Evaluation Campaign of Natural Language Processing and Speech Tools for Italian. Final Workshop (EVALITA 2020)},
    editor = {Basile, Valerio and Croce, Danilo and Di Maro, Maria, and Passaro, Lucia C.},
    publisher = {CEUR.org},
    year = {2020},
    address = {Online}
}

@inproceedings{sarti-etal-2021-looks,
    title = "That Looks Hard: Characterizing Linguistic Complexity in Humans and Language Models",
    author = "Sarti, Gabriele and
    Brunato, Dominique and
    Dell'Orletta, Felice",
    booktitle = "Proceedings of the Workshop on Cognitive Modeling and Computational Linguistics",
    month = jun,
    year = "2021",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "TBD",
    doi = "TBD",
    pages = "TBD",
}
```

## Overview

⚠️ TODO: Short summary and images ⚠️

## Installation

**Prerequisites**

- Python >= 3.6 is required to run the scripts provided in this repository. Torch should be installed using the wheels available on the Pytorch website that are compatible with your CUDA version.

- For CUDA 10 and Python 3.6, we used the wheel torch-1.3.0-cp36-cp36m-linux_x86_64.whl.

- Python >= 3.7 is required to run SyntaxGym-related scripts.

**Main dependencies**

- `torch == 1.6.0`
- `farm == 0.5.0`
- `transformers == 3.3.1`
- `syntaxgym`

**Setup procedure**

```shell
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
./scripts/setup.sh
```

Run `scripts/setup.sh` from the main project folder. This will install dependencies, download data and create the repository structure. If you want to download ZuCo MAT files (30GB), edit `setup.sh` setting `DOWNLOAD_ZUCO_MAT_FILES=false`.

You need to manually download the original perceived complexity dataset presented in [Brunato et al. 2018](https://www.aclweb.org/anthology/D18-1289/) from the [ItaliaNLP Lab website](http://www.italianlp.it/resources/corpus-of-sentences-rated-with-human-complexity-judgments/download-english-sentences/) and place it in the `data/complexity` folder.

The AcCompl-IT campaign data and the Dundee corpus cannot be redistributed due to copyright restrictions.

After all datasets are in the respective folders, run `python script/preprocess.py --all` from the main project folder to preprocess the datasets. Refer to the [Getting Started](#getting-started) section for further steps.

## Code Overview

**Repository structure**

- `data` contains the subfolders for all data used throughout the study:

    - `complexity`: the Perceived Complexity corpus by [Brunato et al. 2018](https://www.aclweb.org/anthology/D18-1289/).
    - `eyetracking`: Eye-tracking corpora (Dundee, GECO, ZuCo 1 & 2).
    - `eval`: SST dataset used for representational similarity evaluation.
    - `garden_paths`: three test suites taken from the [SyntaxGym](syntaxgym.org/) benchmark.
    - `readability`: OneStopEnglish corpus paragraphs by reading level.
    - `preprocessed`: The preprocessed versions of each corpus produced by `scripts/preprocess.py`.

- `src/lingcomp` is the library built behind this work, composed by:
  - `data_utils`: Eye-tracking processors and utils.
  - `farm`: Custom extension of the FARM library to add token-level regression, better multitask learning for NLMs and the GPT-2 model.
  - `similarity`: Methods used for representational similarity evaluation.
  - `syntaxgym`: Methods used to perform evaluation over SyntaxGym test suites.

- `scripts`: Used to carry out the analysis and modeling experiment:
  - `shortcuts`: **in development**, scripts calling other scripts multiple times to provide a quick interface.
  - `analyze_linguistic_features`: Produces a report containing correlations across various complexity metrics and linguistic features.
  - `compute_sentence_baselines`: Computes sentence-level avg., binned avg. and SVM baselines for complexity scores using cross-validation.
  - `compute_similarity`: Evaluates the representational similarity of embeddings produced by neural language models using different methods.
  - `evaluate_garden_paths`: Allows using custom metrics (surprisal, gaze metrics prediction) to estimate the presence of atypical construction over SyntaxGym test suites.
  - `finetune_sentence_level`: Train NLMs on sentence-level regression or classification tasks in single or multi-task settings.
  - `finetune_token_regression`: Train NLMs on token-level regression in single or multi-task settings.
  - `get_surprisals`: Compute surprisal scores produced by NLMs for sentences.
  - `preprocess`: Performs initial preprocessing and train/test splitting.

## Getting Started

**Preprocessing**

```shell
# Generate sentence-level dataset for eyetracking
python scripts/preprocess.py \
    --all \
    --do_features \
    --eyetracking_mode sentence \
    --do_train_test_split
```

⚠️ TODO: Examples for all experiments ⚠️

## Contacts

If you have any questions, feel free to contact me through email ([gabriele.sarti996@gmail.com](mailto:gabriele.sarti996@gmail.com)) or raise a Github issue in the repository!
