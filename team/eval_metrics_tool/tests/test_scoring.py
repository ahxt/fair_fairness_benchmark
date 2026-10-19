from eval_metrics.extraction import PaperContent, Table
from eval_metrics.scoring import WEIGHTS, rank_metrics
from eval_metrics.sections import split_sections


def paper(text, tables=()):
    return PaperContent(sections=split_sections(text), tables=list(tables))


def ranked(text, tables=()):
    return [(s.name, s.score) for s in rank_metrics(paper(text, tables))]


def test_related_work_only_metric_is_excluded():
    text = "Introduction\nWe study things.\nRelated Work\nPrior work reports BLEU and METEOR.\n" \
           "Experiments\nWe measure accuracy.\n"
    names = [n for n, _ in ranked(text)]
    assert "BLEU" not in names and "METEOR" not in names
    assert "Accuracy" in names


def test_references_are_ignored():
    text = "Experiments\nWe measure F1.\nReferences\nA. Author. BLEU: a method. 2002.\n"
    assert [n for n, _ in ranked(text)] == ["F1"]


def test_table_header_outranks_single_intro_mention():
    text = "Introduction\nWe care about BLEU.\n"
    table = Table(rows=[["Model", "F1", "Acc"], ["Ours", "85.2", "90.1"]], section="results")
    result = ranked(text, [table])
    scores = dict(result)
    assert scores["F1"] == WEIGHTS["table_header"] > scores["BLEU"] == WEIGHTS["other"]
    assert [n for n, _ in result] == ["Accuracy", "F1", "BLEU"]  # 5, 5 (alphabetical), 1


def test_weights_by_section():
    text = ("Abstract\nWe report BLEU.\nIntroduction\nWe note CIDEr.\n"
            "Experiments\nWe report WER.\n")
    assert dict(ranked(text)) == {"BLEU": 3, "WER": 2, "CIDEr": 1}


def test_caption_weight():
    text = "Experiments\nTable 1: Results in terms of ROUGE on the test set.\n"
    assert ranked(text) == [("ROUGE", WEIGHTS["caption"])]


def test_false_positive_traps_produce_nothing():
    text = "Experiments\nAs it IS shown below, the R2 region and AP news are irrelevant.\n"
    assert ranked(text) == []


def test_short_acronym_counts_in_table_header_and_next_to_number():
    table = Table(rows=[["Model", "EM", "F1"], ["A", "70.1", "80.2"]], section="results")
    result = dict(ranked("Experiments\nWe get EM = 70.1.\n", [table]))
    assert result["Exact Match"] == WEIGHTS["table_header"] + WEIGHTS["results"]


def test_table_in_related_work_ignored():
    table = Table(rows=[["Model", "BLEU"], ["A", "30.1"]], section="related_work")
    assert ranked("Related Work\nstuff\n", [table]) == []


def test_multi_row_header_and_transposed_table():
    multi = Table(rows=[["", "Dev", "Test"], ["Model", "MAE", "RMSE"], ["A", "1.0", "2.0"]])
    assert dict(ranked("", [multi])) == {"MAE": 5, "RMSE": 5}
    transposed = Table(rows=[["Metric", "Ours"], ["BLEU", "30.1"], ["CIDEr", "90.2"]])
    assert dict(ranked("", [transposed])) == {"BLEU": 5, "CIDEr": 5}


def test_ties_broken_alphabetically_and_ordering():
    text = "Experiments\nWe report WER, BLEU and CER. We also report BLEU.\n"
    assert [n for n, _ in ranked(text)] == ["BLEU", "CER", "WER"]


def test_no_headings_falls_back_to_other():
    assert ranked("We report accuracy.") == [("Accuracy", WEIGHTS["other"])]


def test_empty():
    assert ranked("") == []


def test_section_detection_variants():
    text = ("Title\nAbstract\nx\n1 Introduction\ny\n2. RELATED WORK\nz\n"
            "4.1 Results and Discussion\nr\n5 Conclusion\nc\nReferences\nq\nAppendix A\nap\n")
    assert [s.kind for s in split_sections(text)] == [
        "other", "abstract", "introduction", "related_work", "results",
        "conclusion", "references", "other"]
    assert split_sections("Abstract—We propose X.")[0].kind == "abstract"
