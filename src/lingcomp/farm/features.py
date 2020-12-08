import torch
from farm.data_handler.samples import Sample
from farm.modeling.prediction_head import RegressionHead


class FeaturesEmbeddingSample(Sample):
    def __init__(self, id, clear_text, tokenized=None, features=None, feat_embeds=None):
        super().__init__(id, clear_text, tokenized, features)
        self.feats_embed = feat_embeds


class FeaturesRegressionHead(RegressionHead):
    """A regression head mixing [CLS] representation
    and explicit features for prediction"""

    def forward(self, x, feats, **kwargs):
        x = torch.cat((x, feats), 1)
        logits = self.feed_forward(x)
        return logits
