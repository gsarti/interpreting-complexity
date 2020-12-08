import logging
import os

import torch
from farm.modeling.prediction_head import FeedForwardBlock, PredictionHead
from torch.nn import MSELoss
from torch.nn.functional import pad

from lingcomp.farm.utils import roll


logger = logging.getLogger(__name__)


# TokenRegressionHead
class TokenRegressionHead(PredictionHead):
    def __init__(self, layer_dims=[768, 1], task_name="token_regression", spillover=0, mask_cls=True, **kwargs):
        """
        :param layer_dims: The size of the layers in the feed forward component. The feed forward will have as many layers as there are ints in this list.
        :type layer_dims: list
        :param task_name:
        :param spillover: If > 0, token values are summed with a kernel of this size before being passed to the feedforward layer.
        :param mask_cls: If spillover is specified, defines if the initial token should be masked or not during averaging.
        :param kwargs:
        """
        super(TokenRegressionHead, self).__init__()
        # num_labels could in most cases also be automatically retrieved from the data processor
        self.layer_dims = layer_dims
        # num_labels is being set to 2 since it is being hijacked to store the scaling factor and the mean
        self.num_labels = 2
        if spillover > 0:
            logger.info(f"Spillover mode with size {spillover}, mask_cls: {mask_cls}")
            logger.info(
                f"Prediction head initialized with size [{self.layer_dims[0]} * {(spillover + 1)}, {self.layer_dims[1]}]"
            )
            self.feed_forward = FeedForwardBlock([self.layer_dims[0] * (spillover + 1), self.layer_dims[1]])
        else:
            logger.info(f"Prediction head initialized with size {self.layer_dims}")
            self.feed_forward = FeedForwardBlock(self.layer_dims)
        self.loss_fct = MSELoss(reduction="none")
        self.ph_output_type = "per_token"
        self.model_type = "token_regression"
        self.task_name = task_name
        self.spillover = spillover
        self.mask_cls = mask_cls
        self.generate_config()

    @classmethod
    def load(cls, pretrained_model_name_or_path):
        """
        Load a prediction head from a saved FARM or transformers model. `pretrained_model_name_or_path`
        can be one of the following:
        a) Local path to a FARM prediction head config (e.g. my-bert/prediction_head_0_config.json)
        :param pretrained_model_name_or_path: local path of a saved model or name of a publicly available model.
                                              See https://huggingface.co/models for full list
        """
        if (
            os.path.exists(pretrained_model_name_or_path)
            and "config.json" in pretrained_model_name_or_path
            and "prediction_head" in pretrained_model_name_or_path
        ):
            # a) FARM style
            head = super(TokenRegressionHead, cls).load(pretrained_model_name_or_path)
        else:
            raise NotImplementedError("Load from Transformers not supported yet.")
        return head

    def forward(self, X):
        if self.spillover > 0:
            if self.mask_cls:
                # Create mask on [CLS]
                cls_mask = torch.ones(X.size()).bool()
                cls_mask[:, 0, :] = False
                # Apply mask
                # [batch, seq_len, hidden] => [batch, seq_len - 1, hidden]
                m = X[cls_mask].reshape(X.shape[0], X.shape[1] - 1, X.shape[2])
                # Rolling concat of embeddings for spillover tokens
                # [batch, seq_len - 1, hidden] => [batch, seq_len - 1, hidden * (spillover + 1)]
                out = torch.cat([roll(m, shift, 1, 0) for shift in range(self.spillover, -1, -1)], dim=2)
                ret = torch.cat((pad(X[:, 0, :].unsqueeze(1), (0, out.shape[2] - X.shape[2], 0, 0)), out), dim=1)
            else:
                # Rolling sum of the unmasked sequence
                ret = torch.cat([roll(X, shift, 1, 0) for shift in range(self.spillover, -1, -1)], dim=2)
            logits = self.feed_forward(ret)
        else:
            logits = self.feed_forward(X)
        return logits

    def logits_to_loss(self, logits, initial_mask, padding_mask=None, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        label_ids = label_ids.float()

        # masking on padding and non-initial tokens
        active_loss = (padding_mask.view(-1) == 1) & (initial_mask.view(-1) == 1)

        active_logits = logits.view(-1)[active_loss]
        active_labels = label_ids.view(-1)[active_loss]
        loss = self.loss_fct(active_logits, active_labels)  # loss is a 1 dimensional (active) token loss
        return loss

    def logits_to_preds(self, logits, initial_mask, **kwargs):
        preds_token = logits.detach().cpu().numpy()
        initial_mask = initial_mask.detach().cpu().numpy()
        preds_word_all = []
        for preds_token_one_sample, initial_mask_one_sample in zip(preds_token, initial_mask):
            # Get labels and predictions for just the word initial tokens
            preds_word_id = self.initial_token_only(preds_token_one_sample, initial_mask=initial_mask_one_sample)
            # Rescaling predictions to actual label distributions
            preds_word = [x[0] * self.label_list[1] + self.label_list[0] for x in preds_word_id]
            preds_word_all.append(preds_word)
        return preds_word_all

    def prepare_labels(self, initial_mask, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        label_ids = label_ids.cpu().numpy()
        initial_mask = initial_mask.detach().cpu().numpy()
        labels_all = []
        for label_ids_one_sample, initial_mask_one_sample in zip(label_ids, initial_mask):
            label_ids = self.initial_token_only(label_ids_one_sample, initial_mask=initial_mask_one_sample)
            labels = [x * self.label_list[1] + self.label_list[0] for x in label_ids]
            labels_all.append(labels)
        return labels_all

    @staticmethod
    def initial_token_only(seq, initial_mask):
        ret = []
        for init, s in zip(initial_mask, seq):
            if init:
                ret.append(s)
        return ret

    def formatted_preds(self, logits, initial_mask, samples, **kwargs):
        preds = self.logits_to_preds(logits, initial_mask)

        # align back with original input by getting the original word spans
        spans = []
        for sample, _ in zip(samples, preds):
            word_spans = []
            span = None
            for token, offset, start_of_word in zip(
                sample.tokenized["tokens"], sample.tokenized["offsets"], sample.tokenized["start_of_word"],
            ):
                if start_of_word:
                    # previous word has ended unless it's the very first word
                    if span is not None:
                        word_spans.append(span)
                    span = {"start": offset, "end": offset + len(token)}
                else:
                    # expand the span to include the subword-token
                    span["end"] = offset + len(token.replace("##", ""))
            word_spans.append(span)
            spans.append(word_spans)

        assert len(preds) == len(spans)

        res = {"task": self.task_name, "predictions": []}
        for preds_seq, sample, spans_seq in zip(preds, samples, spans):
            seq_res = []
            for score, span in zip(preds_seq, spans_seq):
                context = sample.clear_text["text"][span["start"] : span["end"]]
                seq_res.append(
                    {
                        "start": span["start"],
                        "end": span["end"],
                        "context": f"{context}",
                        f"{self.task_name}_score": f"{score}",
                    }
                )
            res["predictions"].append(seq_res)
        return res
