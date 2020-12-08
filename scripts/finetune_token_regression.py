import argparse
import logging
import os

import pandas as pd
from farm.data_handler.data_silo import DataSilo, DataSiloForCrossVal
from farm.modeling.optimization import initialize_optimizer
from farm.train import EarlyStopping
from farm.utils import MLFlowLogger, initialize_device_settings, set_all_seeds
from torch.multiprocessing import set_sharing_strategy

from lingcomp.farm.language_model import CustomAdaptiveModel, CustomLanguageModel
from lingcomp.farm.multitask import MultitaskEvaluator, MultitaskInferencer, MultitaskTrainer
from lingcomp.farm.prediction_head import TokenRegressionHead
from lingcomp.farm.processor import TokenRegressionProcessor
from lingcomp.farm.tokenization import CustomTokenizer
from lingcomp.metrics import token_level_regression_metrics
from lingcomp.script_utils import compute_weighted_loss, save_tsv


# Needed to avoid crashing the DataLoader for heavy MTL
set_sharing_strategy("file_system")


# Steps run for each CV fold
def train_on_split(args, silo, processor, fold=None):
    if args.folds > 1:
        args.logger.info(f"############ Crossvalidation: Fold {fold} ############")
    language_model = CustomLanguageModel.load(args.model_name, language_model_class=args.model_class_name)
    if args.prediction_layer > 0:
        language_model.enable_hidden_states_output()
    prediction_heads = [
        TokenRegressionHead(
            layer_dims=[args.heads_dim, 1], task_name=task, spillover=args.spillover, mask_cls=args.no_mask_cls
        )
        for task in args.label_columns
    ]
    # Sum all by default
    loss_fct = None if args.task_weights is None else compute_weighted_loss(args.task_weights, args.label_columns)
    out_types = ["per_token" for _ in args.label_columns]
    # Create an AdaptiveModel = LM + prediction head(s)
    model = CustomAdaptiveModel(
        language_model=language_model,
        prediction_heads=prediction_heads,
        embeds_dropout_prob=args.embed_dropout_prob,
        lm_output_types=out_types,
        device=args.device,
        loss_aggregation_fn=loss_fct,
        freeze_model=args.freeze_model,
        prediction_layer=args.prediction_layer,
    )
    # Create an optimizer
    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=args.learning_rate,
        n_batches=len(silo.loaders["train"]),
        n_epochs=args.num_train_epochs,
        device=args.device,
        grad_acc_steps=args.grad_acc_steps,
    )
    # Setup early stopping
    earlystopping = None
    fold_save_dir = f"{args.save_dir}_{fold}" if args.folds > 1 else args.save_dir
    if args.patience is not None:
        earlystopping = EarlyStopping(metric="loss", mode="min", save_dir=fold_save_dir, patience=args.patience)
    # Feed everything to the trainer
    trainer = MultitaskTrainer(
        model=model,
        optimizer=optimizer,
        data_silo=silo,
        epochs=args.num_train_epochs,
        n_gpu=args.n_gpu,
        device=args.device,
        lr_schedule=lr_schedule,
        evaluate_every=args.evaluate_every,
        early_stopping=earlystopping,
        evaluator_test=False,
        eval_report=False,
    )
    # Let it grow
    trainer.train()
    if args.patience is None:
        # Store the model, only if it wasn't already saved by early stopping
        model.save(fold_save_dir)
        processor.save(fold_save_dir)
    return trainer.model


def evaluate_kfold(args, data_silo, processor):
    silos = DataSiloForCrossVal.make(data_silo, n_splits=args.folds)
    # Run the whole training, earlystopping to get a model,
    # then evaluate the model on the test set of each fold
    dict_preds_labels = {}
    for task in args.label_columns:
        dict_preds_labels[task] = {}
        dict_preds_labels[task]["preds"], dict_preds_labels[task]["labels"] = [], []
    for num_fold, silo in enumerate(silos):
        if not args.do_eval_only:
            model = train_on_split(args, silo, processor, num_fold)
        else:
            model = CustomAdaptiveModel.load(f"{args.model_name}_{num_fold}", device=args.device,)
            model.connect_heads_with_processor(silo.processor.tasks, require_labels=True)
        evaluator_test = MultitaskEvaluator(
            data_loader=silo.get_data_loader("test"), tasks=silo.processor.tasks, device=args.device, report=False
        )
        result = evaluator_test.eval(model, return_preds_and_labels=True)
        evaluator_test.log_results(result, "Test", steps=len(silo.get_data_loader("test")), num_fold=num_fold)
        # Exclude total loss
        for res in result[1:]:
            dict_preds_labels[res["task_name"]]["preds"].extend(res.get("preds"))
            dict_preds_labels[res["task_name"]]["labels"].extend(res.get("labels"))
        if args.save_predictions:
            pred_tsv = pd.DataFrame()
            for res in result[1:]:
                pred_tsv[f"{res['task_name']}_preds"] = res.get("preds")
                pred_tsv[f"{res['task_name']}_labels"] = res.get("labels")
            save_tsv(pred_tsv, os.path.join(args.out_dir, f"{args.run_name}_{num_fold}.tsv"))
    args.logger.info("Final results:")
    for task_name, task in dict_preds_labels.items():
        args.logger.info(f"__{task_name}__")
        metrics = token_level_regression_metrics(task["preds"], task["labels"])
        for metric in metrics.keys():
            args.logger.info(f"{metric}: {metrics[metric]}")


def finetune_token_regression(args):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s  %(message)s", datefmt="%d-%m-%y %H:%M:%S", level=logging.INFO
    )
    args.logger = logging.getLogger(__name__)
    if args.do_logfile:
        filehandler = logging.FileHandler(os.path.join(args.log_dir, f"{args.run_name}.log"))
        args.logger.addHandler(filehandler)
    args.logger.info(vars(args))
    # Setup MLFlow
    ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    ml_logger.init_experiment(experiment_name=args.experiment_name, run_name=args.run_name)
    set_all_seeds(seed=args.seed)
    args.device, args.n_gpu = initialize_device_settings(use_cuda=True)
    # Create a tokenizer
    tok_class = None if not args.model_class_name else f"{args.model_class_name}Tokenizer"
    tokenizer = CustomTokenizer.load(
        pretrained_model_name_or_path=args.model_name, do_lower_case=args.do_lower_case, tokenizer_class=tok_class
    )
    # Create a processor for the dataset
    # Only token-level regression supported for now
    processor = TokenRegressionProcessor(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_length,
        data_dir=args.data_dir,
        label_column_names=args.label_columns,
        label_names=args.label_columns,
        dev_split=args.dev_split,
    )
    # Create a DataSilo that loads several datasets (train/dev/test)
    # provides DataLoaders and calculates descriptive statistics
    data_silo = DataSilo(processor=processor, batch_size=args.batch_size)
    if args.folds > 1:
        evaluate_kfold(args, data_silo, processor)
    else:
        adapt_model = train_on_split(args, data_silo, processor)
        evaluator_test = MultitaskEvaluator(
            data_loader=data_silo.get_data_loader("test"),
            tasks=data_silo.processor.tasks,
            device=args.device,
            report=False,
        )
        result = evaluator_test.eval(adapt_model, return_preds_and_labels=True)
        evaluator_test.log_results(result, "Test", steps=len(data_silo.get_data_loader("test")))
        pred_tsv = pd.DataFrame()
        args.logger.info("Test results:")
        for res in result[1:]:
            args.logger.info(f"__{res['task_name']}__")
            metrics = token_level_regression_metrics(res.get("preds"), res.get("labels"))
            for metric in metrics.keys():
                args.logger.info(f"{metric}: {metrics[metric]}")
            if args.save_predictions:
                pred_tsv[f"{res['task_name']}_preds"] = res.get("preds")
                pred_tsv[f"{res['task_name']}_labels"] = res.get("labels")
        if args.save_predictions:
            save_tsv(pred_tsv, os.path.join(args.out_dir, f"{args.run_name}.tsv"))
        # Load trained model and perform inference
        dicts = [
            {"text": "The intense interest aroused in the public has now somewhat subsided."},
            {"text": "The quick brown fox jumped over the lazy dog."},
        ]
        model = MultitaskInferencer.load(args.save_dir, gpu=True, level="token")
        result = model.inference_from_dicts(dicts=dicts)
        args.logger.info("Inference example:")
        args.logger.info(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name", default=None, type=str, required=True, help="The name of the experiment to be saved in MLflow"
    )
    parser.add_argument(
        "--data_dir",
        default="data/eyetracking/train_test",
        type=str,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--save_dir",
        default="models/bert-eyetracking",
        type=str,
        help="The output dir. Model and processor will be saved there.",
    )
    parser.add_argument("--log_dir", default="logs", type=str, help="The log dir. Logs of runs will be saved there.")
    parser.add_argument("--out_dir", default="out", type=str, help="The output dir. Predictions will be saved there.")
    parser.add_argument(
        "--model_name",
        default="bert-base-cased",
        type=str,
        help="Path to pre-trained model or shortcut name selected from HuggingFace AutoModels",
    )
    parser.add_argument(
        "--model_class_name",
        default=None,
        type=str,
        help="Name of the model class in Transformers form, used to disambiguate models when needed",
    )
    parser.add_argument(
        "--experiment_name",
        default="lingcomp_training",
        type=str,
        help="The name of the experiment to be saved in MLflow",
    )
    parser.add_argument("--folds", type=int, default=1, help="Number of Cross-Validation Folds")
    parser.add_argument(
        "--label_columns",
        nargs="+",
        required=True,
        help="Label column(s) to be used as targets for training."
        "If multiple label columns are specified, the model is trained in multitask mode."
        "Currently cross-modality multitask (classification/regression) is not supported.",
    )
    parser.add_argument(
        "--task_weights",
        nargs="+",
        default=None,
        type=float,
        help="Specifies how the various tasks should be weighted in learning"
        "By default all tasks are weighted equally. Specify one value for each task in label_columns",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--heads_dim", default=768, type=int, help="Model head size (default %(default)s for base models)"
    )
    parser.add_argument("--grad_acc_steps", default=1, type=int, help="Number of gradient accumulation steps.")
    parser.add_argument(
        "--evaluate_every",
        type=int,
        default=100,
        help="After how many steps the model should be evaluated. Set 0 for no evaluation during training.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--embed_dropout_prob", default=0.1, type=float, help="Dropout probability for embeddings.")
    parser.add_argument("--dev_split", default=0.1, type=float, help="Development split provided to data processor.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--patience", default=None, type=int, help="Patience for early stopping.")
    parser.add_argument(
        "--do_logfile", action="store_true", help="Set this flag if you want to log the current run to a file."
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Set this flag if you want to save labels and predictions in a tsv file.",
    )
    parser.add_argument(
        "--freeze_model",
        action="store_true",
        help="If specified, the model is frozen and only prediction head parameters can vary."
        "Use this for probing representations with diagnostic classifiers.",
    )
    parser.add_argument(
        "--do_eval_only", action="store_true", help="If specified, the model is only evaluated without training.",
    )
    parser.add_argument(
        "--spillover",
        default=0,
        type=int,
        help="If specified > 0, simulates spillover effects by"
        "performing a rolling sum with kernel N of token embeddings before the forward pass. Off by default.",
    )
    parser.add_argument(
        "--no_mask_cls",
        action="store_false",
        help="Needs to be passed only if we're using spillover with a model"
        "that doesn't have a pooled representation (e.g. BERT does not need it since it has [CLS], GPT-2 does).",
    )
    parser.add_argument(
        "--prediction_layer",
        default=-1,
        type=int,
        help="Specifies the model layer used for prediction. Default is %(default)s, aka last layer of the model."
        "1 corresponds to first layer after embeddings, 12 to last layer before prediction head in base models."
        "Can be used alongside freeze_model to perform probing tasks on all model layers.",
    )
    args = parser.parse_args()
    if args.save_predictions and not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    finetune_token_regression(args)


if __name__ == "__main__":
    main()
