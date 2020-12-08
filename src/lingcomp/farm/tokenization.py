import logging

from farm.modeling.tokenization import Tokenizer
from transformers.tokenization_gpt2 import GPT2Tokenizer


logger = logging.getLogger(__name__)


class CustomTokenizer(Tokenizer):
    """ Add custom models for tokenization """

    def __init__(self):
        super(CustomTokenizer, self).__init__()

    @classmethod
    def load(cls, pretrained_model_name_or_path, tokenizer_class=None, **kwargs):
        try:
            ret = super(CustomTokenizer, cls).load(pretrained_model_name_or_path, tokenizer_class, **kwargs)
            return ret
        except:  # Custom models
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            if tokenizer_class is None:
                if "gpt2" in pretrained_model_name_or_path.lower():
                    tokenizer_class = "GPT2Tokenizer"
                else:
                    raise ValueError(f"Could not infer tokenizer_class from model config or "
                             f"name '{pretrained_model_name_or_path}'. Set arg `tokenizer_class` "
                             f"in Tokenizer.load() to one of: AlbertTokenizer, XLMRobertaTokenizer, "
                             f"RobertaTokenizer, DistilBertTokenizer, BertTokenizer, XLNetTokenizer, "
                             f"CamembertTokenizer, ElectraTokenizer, DPRQuestionEncoderTokenizer,"
                             f"DPRContextEncoderTokenizer.")
                logger.info(f"Loading tokenizer of type '{tokenizer_class}'")
            ret = None
            if tokenizer_class == "GPT2Tokenizer":
                ret = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
                ret.pad_token = ret.unk_token
            if ret is None:
                raise Exception("Unable to load tokenizer")
            else:
                return ret
