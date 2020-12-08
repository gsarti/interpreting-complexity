import json

from syntaxgym.prediction import Prediction
from syntaxgym.suite import Suite


def custom_load_suite(suite_ref):
    if isinstance(suite_ref, CustomSuite):
        return suite_ref

    # Load from dict / JSON file / JSON path
    if not isinstance(suite_ref, dict):
        if not hasattr(suite_ref, "read"):
            suite_ref = open(suite_ref, "r")
        suite = json.load(suite_ref)
    else:
        suite = suite_ref
    return CustomSuite.from_dict(suite)


class CustomPrediction(Prediction):
    """ Customize for multiple sub-metric values """

    def __call__(self, item, score):
        scores = {}
        for c in item["conditions"]:
            for r in c["regions"]:
                if "metric_value" in r:
                    scores[c["condition_name"], r["region_number"]] = r["metric_value"][score]["sum"]
        return self.formula(scores)


class CustomSuite(Suite):
    """ Allows for multiple score predictions """

    def __init__(self, condition_names, region_names, items, predictions, meta):
        super().__init__(condition_names, region_names, items, predictions, meta)
        self.score_columns = []

    @classmethod
    def from_dict(cls, suite_dict):
        base_suite = super().from_dict(suite_dict)
        preds = [CustomPrediction.from_dict(pred_i, i) for i, pred_i in enumerate(suite_dict["predictions"])]
        return cls(
            condition_names=base_suite.condition_names,
            region_names=base_suite.region_names,
            items=base_suite.items,
            predictions=preds,
            meta=suite_dict["meta"],
        )

    def evaluate_predictions(self):
        """
        Compute prediction results for each item.
        Returns:
            results: a nested dict mapping ``(item_number => score_column =>
                prediction =>  prediction_result)``
        """
        result = {}
        for item in self.items:
            result[item["item_number"]] = {}
            for score in self.score_columns:
                result[item["item_number"]][score] = {}
                for pred in self.predictions:
                    result[item["item_number"]][score][pred] = pred(item, score)
        return result
