import json
import logging
import os
from pathlib import Path

from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import PredictionHead
from transformers import GPT2Config, GPT2Model
from transformers.modeling_auto import AutoConfig
from transformers.modeling_utils import SequenceSummary


logger = logging.getLogger(__name__)


class CustomLanguageModel(LanguageModel):
    """
    Custom language model using kwargs in the initialization.
    """

    @classmethod
    def load(cls, pretrained_model_name_or_path, n_added_tokens=0, language_model_class=None, **kwargs):
        """
        Allows for loading local config with kwargs
        Do not support custom vocabularies
        """
        config_file = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(config_file):
            # it's a local directory in FARM format
            config = json.load(open(config_file))
            kwargs.pop("farm_lm_name", None)
            language_model = cls.subclasses[config["name"]].load(pretrained_model_name_or_path, **kwargs)
            language_model.pooling_strategy = config.get("summary_type", None)
        else:
            if language_model_class is None:
                language_model_class = cls.get_language_model_class(pretrained_model_name_or_path)
                # Custom models to check if class is still None
                if language_model_class is None:
                    language_model_class = cls.get_custom_language_model_class(pretrained_model_name_or_path)
         
            if language_model_class:
                language_model = cls.subclasses[language_model_class].load(pretrained_model_name_or_path, **kwargs)
            else:
                raise Exception(
                    f"Model not found for {pretrained_model_name_or_path}. Either supply the local path for a saved "
                    f"model or one of bert/roberta/xlnet/albert/distilbert models that can be downloaded from remote. "
                    f"Ensure that the model class name can be inferred from the directory name when loading a "
                    f"Transformers' model. Here's a list of available models: "
                    f"https://farm.deepset.ai/api/modeling.html#farm.modeling.language_model.LanguageModel.load"
                )

            # Resize embeddings in case of custom vocab
            if n_added_tokens != 0:
                model_emb_size = language_model.model.resize_token_embeddings(new_num_tokens=None).num_embeddings
                vocab_size = model_emb_size + n_added_tokens
                logger.info(
                    f"Resizing embedding layer of LM from {model_emb_size} to {vocab_size} to cope with custom vocab."
                )
                language_model.model.resize_token_embeddings(vocab_size)
                # verify
                model_emb_size = language_model.model.resize_token_embeddings(new_num_tokens=None).num_embeddings
                assert vocab_size == model_emb_size, "Vocabulary and model embedding sizes do not match"
        return language_model
    
    @staticmethod
    def get_custom_language_model_class(model_name_or_path):
        # it's transformers format (either from model hub or local)
        model_name_or_path = str(model_name_or_path)
        config = AutoConfig.from_pretrained(model_name_or_path)
        model_type = config.model_type
        if model_type == "gpt2":
            language_model_class = "GPT2"
        else:
            language_model_class = CustomLanguageModel._infer_custom_language_model_class_from_string(model_name_or_path)
        return language_model_class

    @staticmethod
    def _infer_custom_language_model_class_from_string(model_name_or_path):
        if "gpt2" in model_name_or_path.lower():
            language_model_class = "GPT2"
        else:
            language_model_class = None
        return language_model_class



class CustomAdaptiveModel(AdaptiveModel):
    """
    head_feats: if true the model is expected to receive extra explicit features
    during training. Those will be concat to sentence representations and used for prediction.
    freeze_model: freeze LM weights, allowing only for prediction head training, can be used for probing.
    custom_pooling: Uses options provided by transformer's SequenceSummary to aggregate embeddings for sentence-level tasks
    prediction_layer: Specifies which layer should be used for prediction, used for probing tasks.
    """

    def __init__(
        self,
        language_model,
        prediction_heads,
        embeds_dropout_prob,
        lm_output_types,
        device,
        loss_aggregation_fn=None,
        head_feats=False,
        freeze_model=False,
        custom_pooling_strategy=None,
        prediction_layer=-1,
    ):
        self.head_feats = head_feats
        self.pooler = None
        super(CustomAdaptiveModel, self).__init__(
            language_model, prediction_heads, embeds_dropout_prob, lm_output_types, device, loss_aggregation_fn
        )
        if freeze_model:
            for p in self.language_model.parameters():
                p.requires_grad = False
        if custom_pooling_strategy is not None:
            config = self.language_model.model.config
            config.summary_type = custom_pooling_strategy
            self.pooler = SequenceSummary(config)
            self.pooler.apply(self.language_model.model._init_weights)
            logger.info(f"Using custom pooling strategy: {custom_pooling_strategy}")
        self.prediction_layer = prediction_layer

    @classmethod
    def load(cls, load_dir, device, strict=True, lm_name=None, processor=None, **kwargs):
        """
        Allows for passing custom kwargs to language model and for custom loss_aggregation_fn
        to keep same training setup after reloading.
        """
        # Language Model
        if lm_name:
            language_model = CustomLanguageModel.load(load_dir, farm_lm_name=lm_name, **kwargs)
        else:
            language_model = CustomLanguageModel.load(load_dir, **kwargs)

        # Prediction heads
        _, ph_config_files = cls._get_prediction_head_files(load_dir)
        prediction_heads = []
        ph_output_type = []
        for config_file in ph_config_files:
            head = PredictionHead.load(config_file, strict=strict)
            prediction_heads.append(head)
            ph_output_type.append(head.ph_output_type)
        loss_aggregation_fn = kwargs.get("loss_aggregation_fn", None)
        model = cls(
            language_model,
            prediction_heads,
            0.1,
            ph_output_type,
            device,
            loss_aggregation_fn=loss_aggregation_fn,
            custom_pooling_strategy=language_model.pooling_strategy,
        )
        if processor:
            model.connect_heads_with_processor(processor.tasks)
        return model

    def forward(self, **kwargs):
        """
        Allows for passing custom kwargs to heads
        """
        # Run forward pass of language model
        if self.prediction_layer == -1:
            sequence_output, pooled_output = self.forward_lm(**kwargs)
        else:  # We do prediction on a layer which is not the last one
            sequence_output, pooled_output, hidden_states = self.forward_lm(**kwargs)
            sequence_output = hidden_states[self.prediction_layer]
        # The following condition is checked only when we're dealing with sentence-level tasks that require pooling
        # In this way fine-tuning for token regression doesn't require setting a pooler
        if (
            self.pooler is None
            and self.prediction_layer != -1
            and any(out in ["per_sequence", "per_sequence_continuous"] for out in self.lm_output_types)
        ):
            raise AttributeError("Pooling strategy must be specified when predicting from custom layers.")
        # If custom pooling strategy is specified, replace pooled output with new one
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output)
        # Run forward pass of (multiple) prediction heads using the output from above
        all_logits = []
        if len(self.prediction_heads) > 0:
            for head, lm_out in zip(self.prediction_heads, self.lm_output_types):
                # Choose relevant vectors from LM as output and perform dropout
                if lm_out == "per_token":
                    output = self.dropout(sequence_output)
                elif lm_out == "per_sequence" or lm_out == "per_sequence_continuous":
                    output = self.dropout(pooled_output)
                elif (
                    lm_out == "per_token_squad"
                ):  # we need a per_token_squad because of variable metric computation later on...
                    output = self.dropout(sequence_output)
                else:
                    raise ValueError("Unknown extraction strategy from language model: {}".format(lm_out))
                if self.head_feats:
                    # Do the actual forward pass of a single head
                    all_logits.append(head(output, **kwargs))
                else:
                    all_logits.append(head(output))
        else:
            # just return LM output (e.g. useful for extracting embeddings at inference time)
            all_logits.append((sequence_output, pooled_output))

        return all_logits

    def fit_heads_to_lm(self):
        """ Skip resizing heads to handle the presence of extra features or spillover """
        for ph in self.prediction_heads:
            if not self.head_feats and (not hasattr(ph, "spillover") or ph.spillover < 1):
                ph.resize_input(self.lm_output_dims)
            ph.to(self.device)

    def formatted_preds(self, logits, **kwargs):
        """
        Format predictions for inference.
        :param logits: model logits
        :type logits: torch.tensor
        :param label_maps: dictionary for mapping ids to label strings
        :type label_maps: dict[int:str]
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        :return: predictions in the right format
        """
        all_preds = []
        n_heads = len(self.prediction_heads)
        if n_heads == 0:
            # just return LM output (e.g. useful for extracting embeddings at inference time)
            preds_final = self.language_model.formatted_preds(logits=logits, **kwargs)
        else:
            # collect preds from all heads
            for head, logits_for_head in zip(self.prediction_heads, logits):
                preds = head.formatted_preds(logits=logits_for_head, **kwargs)
                all_preds.append(preds)
            return all_preds


class GPT2(LanguageModel):
    """
    A GPT2 model that wraps HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    """

    def __init__(self):
        super(GPT2, self).__init__()
        self.model = None
        self.name = "gpt2"

    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a pretrained model by supplying
        * the name of a remote model on s3 ("gpt2" ...)
        * OR a local path of a model trained via transformers ("some_dir/huggingface_model")
        * OR a local path of a model trained via FARM ("some_dir/farm_model")
        :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
        :type pretrained_model_name_or_path: str
        """

        gpt2 = cls()
        if "farm_lm_name" in kwargs:
            gpt2.name = kwargs["farm_lm_name"]
        else:
            gpt2.name = pretrained_model_name_or_path
        # We need to differentiate between loading model using FARM format and Pytorch-Transformers format
        farm_lm_config = Path(pretrained_model_name_or_path) / "language_model_config.json"
        if os.path.exists(farm_lm_config):
            # FARM style
            gpt2_config = GPT2Config.from_pretrained(farm_lm_config)
            farm_lm_model = Path(pretrained_model_name_or_path) / "language_model.bin"
            gpt2.model = GPT2Model.from_pretrained(farm_lm_model, config=gpt2_config, **kwargs)
            gpt2.language = gpt2.model.config.language
        else:
            # Pytorch-transformer Style
            gpt2.model = GPT2Model.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            gpt2.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
        return gpt2

    def forward(
        self, input_ids, segment_ids, padding_mask, **kwargs,
    ):
        """
        Perform the forward pass of the GPT2 model.
        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :type input_ids: torch.Tensor
        :param segment_ids: The id of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and those in the second are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len]
        :type segment_ids: torch.Tensor
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.
        """
        output_tuple = self.model(
            input_ids, token_type_ids=segment_ids, attention_mask=padding_mask,
        )  # last hidden state, (presents), (all hidden_states), (attentions)
        if self.model.config.output_hidden_states:
            # GPT2 has no pooled output
            sequence_output, pooled_output, all_hidden_states = output_tuple[0], None, output_tuple[2]
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output, pooled_output = output_tuple[0], None
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.config.output_hidden_states = False
