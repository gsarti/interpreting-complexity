import logging
import os
import sys

import torch
from farm.modeling.adaptive_model import ONNXAdaptiveModel
from farm.modeling.optimization import optimize_model
from farm.data_handler.dataloader import NamedDataLoader
from farm.eval import Evaluator
from farm.infer import Inferencer
from farm.train import Trainer
from farm.utils import initialize_device_settings, set_all_seeds
from farm.visual.ascii.images import GROWING_TREE
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm

from lingcomp.farm.language_model import CustomAdaptiveModel
from lingcomp.farm.processor import CustomProcessor, InferenceProcessor
from lingcomp.farm.utils import format_multitask_preds


logger = logging.getLogger(__name__)


class MultitaskEvaluator(Evaluator):
    """
    Support aggregated loss in eval logging
    This is fundamental for multitask earlystopping
    """

    def eval(self, model, return_preds_and_labels=False, aggregate_fn=None):
        aggregate_fn = sum if aggregate_fn is None else aggregate_fn
        all_results = super().eval(model, return_preds_and_labels)
        # Prepend aggregated loss for all heads
        # Prepending allows using it for earlystopping eval
        all_results.insert(0, {"loss": aggregate_fn([res["loss"] for res in all_results]), "task_name": "total"})
        return all_results


class MultitaskTrainer(Trainer):
    """
    Adapted by using MultitaskEvaluator for training
    """

    def train(self, **kwargs):
        """ kwargs are passed to the language model when it gets restored """
        # connect the prediction heads with the right output from processor
        self.model.connect_heads_with_processor(self.data_silo.processor.tasks, require_labels=True)
        # Check that the tokenizer fits the language model
        #TODO: make this compliant for DP / DDP where the model class is wrapped
        if self.model._get_name() == 'BiAdaptiveModel':
            self.model.verify_vocab_size(vocab_size1=len(self.data_silo.processor.tokenizer),
                                         vocab_size2=len(self.data_silo.processor.passage_tokenizer))
        else:
            self.model.verify_vocab_size(vocab_size=len(self.data_silo.processor.tokenizer))
        self.model.train()
        logger.info(f"Frozen model: {all([p.requires_grad is False for p in self.model.language_model.parameters()])}")
        do_stopping = False
        evalnr = 0
        loss = 0
        resume_from_step = self.from_step

        if self.local_rank in [0, -1]:
            logger.info(f"\n {GROWING_TREE}")

        for epoch in range(self.from_epoch, self.epochs):
            early_break = False
            self.from_epoch = epoch
            train_data_loader = self.data_silo.get_data_loader("train")
            progress_bar = tqdm(train_data_loader, disable=self.local_rank not in [0, -1] or self.disable_tqdm)
            for step, batch in enumerate(progress_bar):
                # when resuming training from a checkpoint, we want to fast forward to the step of the checkpoint
                if resume_from_step and step <= resume_from_step:
                    # TODO: Improve skipping for StreamingDataSilo
                    # The seeds before and within the loop are currently needed, if you need full reproducibility
                    # of runs with vs. without checkpointing using StreamingDataSilo. Reason: While skipping steps in StreamingDataSilo,
                    # we update the state of the random number generator (e.g. due to masking words), which can impact the model behaviour (e.g. dropout)
                    if step % 10000 == 0:
                        logger.info(f"Skipping {step} out of {resume_from_step} steps ...")
                    if resume_from_step == step:
                        logger.info(f"Finished skipping {resume_from_step} steps ...")
                        resume_from_step = None
                    else:
                        continue

                progress_bar.set_description(f"Train epoch {epoch}/{self.epochs-1} (Cur. train loss: {loss:.4f})")

                # Only for distributed training: we need to ensure that all ranks still have a batch left for training
                if self.local_rank != -1:
                    if not self._all_ranks_have_data(has_data=1, step=step):
                        early_break = True
                        break

                # Move batch of samples to device
                batch = {key: batch[key].to(self.device) for key in batch}

                # Forward & backward pass through model
                logits = self.model.forward(**batch)
                per_sample_loss = self.model.logits_to_loss(logits=logits, global_step=self.global_step, **batch)
                loss = self.backward_propagate(per_sample_loss, step)

                # Perform  evaluation
                if (
                    self.evaluate_every != 0
                    and self.global_step % self.evaluate_every == 0
                    and self.global_step != 0
                    and self.local_rank in [0, -1]
                ):
                    # When using StreamingDataSilo, each evaluation creates a new instance of
                    # dev_data_loader. In cases like training from scratch, this could cause
                    # some variance across evaluators due to the randomness in word masking.
                    dev_data_loader = self.data_silo.get_data_loader("dev")
                    if dev_data_loader is not None:
                        evaluator_dev = MultitaskEvaluator(
                            data_loader=dev_data_loader,
                            tasks=self.data_silo.processor.tasks,
                            device=self.device,
                            report=self.eval_report,
                        )
                        evalnr += 1
                        result = evaluator_dev.eval(self.model)
                        evaluator_dev.log_results(result, "Dev", self.global_step)
                        if self.early_stopping:
                            do_stopping, save_model, eval_value = self.early_stopping.check_stopping(result)
                            if save_model:
                                logger.info(
                                    "Saving current best model to {}, eval={}".format(
                                        self.early_stopping.save_dir, eval_value
                                    )
                                )
                                self.model.save(self.early_stopping.save_dir)
                                self.data_silo.processor.save(self.early_stopping.save_dir)
                            if do_stopping:
                                # log the stopping
                                logger.info(
                                    "STOPPING EARLY AT EPOCH {}, STEP {}, EVALUATION {}".format(epoch, step, evalnr)
                                )
                if do_stopping:
                    break

                self.global_step += 1
                self.from_step = step + 1

                # save the current state as a checkpoint before exiting if a SIGTERM signal is received
                if self.sigterm_handler and self.sigterm_handler.kill_now:
                    logger.info("Received a SIGTERM signal. Saving the current train state as a checkpoint ...")
                    if self.local_rank in [0, -1]:
                        self._save()
                        torch.distributed.destroy_process_group()
                        sys.exit(0)

                # save a checkpoint and continue train
                if self.checkpoint_every and step % self.checkpoint_every == 0:
                    if self.local_rank in [0, -1]:
                        self._save()
                    # Let other ranks wait until rank 0 has finished saving
                    if self.local_rank != -1:
                        torch.distributed.barrier()

            if do_stopping:
                break

            # Only for distributed training: we need to ensure that all ranks still have a batch left for training
            if self.local_rank != -1 and not early_break:
                self._all_ranks_have_data(has_data=False)

        # With early stopping we want to restore the best model
        if self.early_stopping and self.early_stopping.save_dir:
            logger.info("Restoring best model so far from {}".format(self.early_stopping.save_dir))
            lm_name = self.model.language_model.name
            model = CustomAdaptiveModel.load(self.early_stopping.save_dir, self.device, lm_name=lm_name, **kwargs)
            model.connect_heads_with_processor(self.data_silo.processor.tasks, require_labels=True)

        # Eval on test set
        if self.evaluator_test:
            test_data_loader = self.data_silo.get_data_loader("test")
            if test_data_loader is not None:
                evaluator_test = MultitaskEvaluator(
                    data_loader=test_data_loader,
                    tasks=self.data_silo.processor.tasks,
                    device=self.device,
                )
                self.test_result = evaluator_test.eval(self.model)
                evaluator_test.log_results(self.test_result, "Test", self.global_step)
        return self.model


class MultitaskInferencer(Inferencer):
    """
    An inferencer specifically tuned to perform multitask token-level regression.
    """

    def __init__(
        self,
        model,
        processor,
        task_type,
        batch_size=4,
        gpu=False,
        name=None,
        return_class_probs=False,
        extraction_strategy=None,
        extraction_layer=None,
        s3e_stats=None,
        num_processes=None,
        disable_tqdm=False,
        benchmarking=False,
        dummy_ph=False,
        level="sentence",
    ):
        super(MultitaskInferencer, self).__init__(
            model,
            processor,
            task_type,
            batch_size,
            gpu,
            name,
            return_class_probs,
            extraction_strategy,
            extraction_layer,
            s3e_stats,
            num_processes,
            disable_tqdm,
            benchmarking,
            dummy_ph,
        )
        self.level = level

    @classmethod
    def load(
        cls,
        model_name_or_path,
        batch_size=4,
        gpu=False,
        task_type=None,
        return_class_probs=False,
        strict=True,
        max_seq_len=256,
        doc_stride=128,
        extraction_layer=None,
        extraction_strategy=None,
        s3e_stats=None,
        num_processes=None,
        disable_tqdm=False,
        tokenizer_class=None,
        use_fast=False,
        tokenizer_args=None,
        dummy_ph=False,
        benchmarking=False,
        level="sentence",
    ):
        """ Load inferencer with CustomAdaptiveModel """
        device, _ = initialize_device_settings(use_cuda=gpu, local_rank=-1, use_amp=None)
        name = os.path.basename(model_name_or_path)
        if os.path.exists(model_name_or_path):
            model = CustomAdaptiveModel.load(load_dir=model_name_or_path, device=device, strict=strict)
            if task_type == "embeddings":
                processor = InferenceProcessor.load_from_dir(model_name_or_path)
            else:
                processor = CustomProcessor.load_from_dir(model_name_or_path)
            processor.max_seq_len = max_seq_len
            if hasattr(processor, "doc_stride"):
                processor.doc_stride = doc_stride
        else:
            logger.info(f"Could not find `{model_name_or_path}` locally. Try to download from model hub ...")
            if not task_type:
                raise ValueError("Please specify the 'task_type' of the model you want to load from transformers. "
                                 "Valid options for arg `task_type`:"
                                 "'question_answering', 'embeddings', 'text_classification', 'ner'")

            model = CustomAdaptiveModel.convert_from_transformers(model_name_or_path, device, task_type)
            processor = CustomProcessor.convert_from_transformers(model_name_or_path, task_type, max_seq_len, doc_stride,
                                                            tokenizer_class, tokenizer_args, use_fast)

        if not isinstance(model,ONNXAdaptiveModel):
            model, _ = optimize_model(model=model, device=device, local_rank=-1, optimizer=None)
        return cls(
            model,
            processor,
            task_type=task_type,
            batch_size=batch_size,
            gpu=gpu,
            name=name,
            return_class_probs=return_class_probs,
            extraction_strategy=extraction_strategy,
            extraction_layer=extraction_layer,
            s3e_stats=s3e_stats,
            num_processes=num_processes,
            disable_tqdm=disable_tqdm,
            benchmarking=benchmarking,
            dummy_ph=dummy_ph,
            level=level,
        )

    def _get_predictions(self, dataset, tensor_names, baskets):
        samples = [s for b in baskets for s in b.samples]

        data_loader = NamedDataLoader(
            dataset=dataset, sampler=SequentialSampler(dataset), batch_size=self.batch_size, tensor_names=tensor_names
        )
        preds_all = []
        for i, batch in enumerate(
            tqdm(data_loader, desc="Inferencing Samples", unit=" Batches", disable=self.disable_tqdm)
        ):
            batch = {key: batch[key].to(self.device) for key in batch}
            batch_samples = samples[i * self.batch_size : (i + 1) * self.batch_size]

            # Two fundamental differences with original:
            # 1) Use all logits for all heads instead of taking only the first one
            # 2) Passes logits as the list that already is and not as element of list
            with torch.no_grad():
                logits = self.model.forward(**batch)
                preds = self.model.formatted_preds(
                    logits=logits,
                    samples=batch_samples,
                    tokenizer=self.processor.tokenizer,
                    return_class_probs=self.return_class_probs,
                    **batch,
                )
            if self.level == "token":
                if i == 0:
                    preds_all = preds
                else:
                    for task_dict in preds:
                        preds_all_dict_id = [
                            idx for idx, dic in enumerate(preds_all) if dic["task"] == task_dict["task"]
                        ]
                        if len(preds_all_dict_id) != 1:
                            raise AttributeError("Task type must be present a single time.")
                        idx = preds_all_dict_id[0]
                        preds_all[idx]["predictions"] += task_dict["predictions"]
            else:
                preds_all += preds
        if self.level == "token":
            preds_all = format_multitask_preds(preds_all)
        return preds_all
