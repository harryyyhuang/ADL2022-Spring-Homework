import datasets
from tw_rouge import get_rouge
from tqdm import tqdm

_CITATION = """
Ta's evaluation
"""

_DESCRIPTION = """
This metrics is created by ta
"""

_KWARGS_DESCRIPTION = """
Not quite sure yet
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Rouge(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
        )

    def _compute(self, predictions, references, rouge_types=None, use_aggregator=True, use_stemmer=False):
        if rouge_types is None:
            rouge_types = ["rouge-1", "rouge-2", "rouge-l"]


        preds = []
        refs = []


        for ref, pred in tqdm(zip(references, predictions), total=len(references)):
            # score = get_rouge(pred, ref)
            preds.append(pred)
            refs.append(ref)
            # scores.append(score)


        # result = {}
        # for key in scores[0].keys():
        #     result[key] = list(score[key] for score in scores)
        # print(preds)

        return get_rouge(preds, refs, ignore_empty=True)