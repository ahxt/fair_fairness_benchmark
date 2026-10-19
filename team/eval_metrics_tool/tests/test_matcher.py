import pytest

from eval_metrics.matcher import find_metrics, normalize_metric
from eval_metrics.metrics_dictionary import METRICS


@pytest.mark.parametrize("alias", ["F-measure", "F1-score", "F1 score", "macro-F1",
                                   "Micro-F1", "F1", "f-score"])
def test_f1_aliases(alias):
    assert normalize_metric(alias) == "F1"


@pytest.mark.parametrize("alias,expected", [
    ("ROUGE-L", "ROUGE"), ("Rouge-1", "ROUGE"), ("ROC-AUC", "AUC"), ("AUROC", "AUC"),
    ("PR-AUC", "PR-AUC"), ("mean average precision", "mAP"), ("mIoU", "IoU"),
    ("Top-1 accuracy", "Top-1 Accuracy"), ("top5 acc.", "Top-5 Accuracy"),
    ("word error rate", "WER"), ("pass@10", "pass@k"), ("R-squared", "R²"),
    ("Recall@10", "Recall@k"), ("Hits@3", "Hits@k"), ("nDCG@10", "NDCG"),
    ("root mean squared error", "RMSE"), ("Fréchet Inception Distance", "FID"),
    ("Pearson's r", "Pearson Correlation"), ("Spearman's rho", "Spearman Correlation"),
    ("chrF++", "chrF"), ("SacreBLEU", "BLEU"), ("EM", "Exact Match"),
])
def test_other_aliases(alias, expected):
    assert normalize_metric(alias) == expected


def test_unknown_and_multiword_strings_are_not_normalized():
    assert normalize_metric("learning rate") is None
    assert normalize_metric("BLEU and ROUGE") is None


def names(text):
    return [m.metric for m in find_metrics(text)]


def test_longest_match_wins():
    assert names("We report Top-1 accuracy.") == ["Top-1 Accuracy"]
    assert names("PR-AUC improves.") == ["PR-AUC"]


def test_short_acronym_traps():
    assert names("As it IS shown in Figure 2, the R2 region is large.") == []
    assert names("The EM algorithm and AP news.") == []


def test_short_acronyms_with_number_context():
    assert names("EM = 85.2 on SQuAD") == ["Exact Match"]
    assert names("an AP of 41.0") == ["mAP"]
    assert names("R2: 0.91") == ["R²"]
    assert names("IS of 9.5") == ["Inception Score"]


def test_short_acronyms_in_table_cells_need_no_number():
    assert [m.metric for m in find_metrics("EM", context_given=True)] == ["Exact Match"]


def test_case_sensitive_acronyms():
    assert names("the mse of the fit") == []
    assert names("MSE dropped") == ["MSE"]


def test_numeric_precision_and_recall_verb_are_ignored():
    assert names("We use mixed precision training. Recall that x holds.") == []
    assert names("Precision and recall are high.") == ["Precision", "Recall"]


def test_dictionary_names_unique_and_patterns_compile():
    assert len({m.name for m in METRICS}) == len(METRICS)
    assert find_metrics("smoke test")  == []
