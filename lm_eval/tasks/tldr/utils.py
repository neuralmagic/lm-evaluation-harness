from rouge_score import rouge_scorer, scoring
from bert_score import BERTScorer
from nltk.translate import meteor
from nltk import word_tokenize
import numpy

import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')

# requirements:
# - pip install rouge-score
# - pip install bert-score
# - pip install hf_xet
# - pip install nltk
# - pip install moverscore
# - pip install pyemd==0.5.1

def process_results(doc, results):
    
    reference = doc["completion"]

    scores = rouge([reference], results)
    scores.update(meteor_score(reference, results[0]))
    scores["bleu"] = [reference, results[0]]
    scores["bert_score"] = [reference, results[0]]
    #scores["mover_score"] = [reference, results[0]]

    return scores


def bert_score(items):
    scorer = BERTScorer(
        lang="en", 
        rescale_with_baseline=True, 
    )
    
    bert_score = []
    for reference, prediction in items:
        _, _, f1 = scorer.score([prediction], [reference])
        bert_score.append(f1.item())


    return numpy.average(bert_score).item()


def meteor_score(target, prediction):
    return {
        "meteor": meteor(
            [word_tokenize(target)], word_tokenize(prediction)
        )
    }


# def mover_score(items):
#     from moverscore_v2 import get_idf_dict, word_mover_score 

#     mover_score = []
#     for reference, prediction in items:

#         idf_dict_hyp = get_idf_dict([prediction])
#         idf_dict_ref = get_idf_dict([reference])

#         mover_score.append(word_mover_score(
#             reference, prediction, 
#             idf_dict_ref, 
#             idf_dict_hyp,
#             stop_words=[], 
#             n_gram=1, 
#             remove_subwords=True,
#         ))

#     return numpy.average(mover_score).item()


def rouge(targets, predictions):
    """Computes rouge score.

    Args:
        targets: list of strings
        predictions: list of strings
        score_keys: list of strings with the keys to compute.
    Returns:
        dict with score_key: rouge score across all targets and predictions
    """

    score_keys = ["rouge1", "rouge2", "rougeLsum"]
    scorer = rouge_scorer.RougeScorer(score_keys)
    aggregator = scoring.BootstrapAggregator()


    def _prepare_summary(summary):
        # Make sure the summary is not bytes-type
        # Add newlines between sentences so that rougeLsum is computed correctly.
        summary = summary.replace(". ", ".\n")
        return summary

    for prediction, target in zip(predictions, targets):
        target = _prepare_summary(target)
        prediction = _prepare_summary(prediction)
        aggregator.add_scores(scorer.score(target=target, prediction=prediction))
    result = aggregator.aggregate()

    return {key: result[key].mid.fmeasure.item() for key in score_keys}