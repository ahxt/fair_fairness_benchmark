import pytest
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors

from eval_metrics import PDFReadError, extract_eval_metrics, rank_pdf


def build_pdf(path):
    st = getSampleStyleSheet()
    h, body = st["Heading2"], st["BodyText"]
    grid = Table([["Model", "F1", "BLEU"], ["Baseline", "70.1", "20.3"], ["Ours", "75.4", "24.8"]])
    grid.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]))
    story = [
        Paragraph("Abstract", h), Paragraph("We propose a model and report F1.", body),
        Paragraph("1 Introduction", h), Paragraph("Machine translation is hard.", body),
        Paragraph("2 Related Work", h), Paragraph("Earlier systems used METEOR and CIDEr.", body),
        Paragraph("3 Experiments", h), Paragraph("Table 1: F1 and BLEU on the test set.", body),
        Spacer(1, 6), grid, Spacer(1, 6),
        Paragraph("Our model improves F1 over the baseline. Accuracy is also given.", body),
        Paragraph("4 Conclusion", h), Paragraph("We are done.", body),
        Paragraph("References", h), Paragraph("[1] A. Author. ROUGE: a package. 2004.", body),
    ]
    SimpleDocTemplate(str(path), pagesize=letter).build(story)


@pytest.fixture
def pdf(tmp_path):
    p = tmp_path / "paper.pdf"
    build_pdf(p)
    return str(p)


@pytest.fixture(autouse=True)
def no_default_model(monkeypatch):
    monkeypatch.delenv("EVAL_METRICS_MODEL", raising=False)


def test_end_to_end_ranking(pdf):
    result = extract_eval_metrics(pdf, top_k=5, use_llm=False)
    assert result[:2] == ["F1", "BLEU"]
    assert "Accuracy" in result
    assert not {"METEOR", "CIDEr", "ROUGE"} & set(result)


def test_tables_were_extracted_and_scored_as_headers(pdf):
    scores = {s.name: s for s in rank_pdf(pdf)}
    assert scores["BLEU"].where["table_header"] == 1
    assert scores["F1"].where["table_header"] == 1


def test_top_k_limits(pdf):
    assert len(extract_eval_metrics(pdf, top_k=1, use_llm=False)) == 1


def test_pdf_without_metrics(tmp_path):
    p = tmp_path / "plain.pdf"
    SimpleDocTemplate(str(p)).build([Paragraph("Nothing to see here.", getSampleStyleSheet()["BodyText"])])
    assert extract_eval_metrics(str(p), use_llm=False) == []


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        extract_eval_metrics("/nonexistent/paper.pdf")


def test_not_a_pdf(tmp_path):
    p = tmp_path / "fake.pdf"
    p.write_text("this is not a pdf")
    with pytest.raises(PDFReadError):
        extract_eval_metrics(str(p))


def test_cli(pdf, capsys):
    from eval_metrics.__main__ import main
    assert main([pdf, "--top-k", "2", "--no-llm"]) == 0
    assert capsys.readouterr().out.split() == ["F1", "BLEU"]
    assert main([pdf, "--verbose", "--top-k", "1", "--no-llm"]) == 0
    assert "score=" in capsys.readouterr().out
    assert main(["/nonexistent.pdf"]) == 1


def test_no_model_warns_and_uses_dictionary(pdf):
    with pytest.warns(UserWarning, match="no LLM model set"):
        assert extract_eval_metrics(pdf, top_k=2) == ["F1", "BLEU"]


def test_llm_path_returns_raw_names(pdf, fake_litellm):
    fake_litellm.reply = '["F1 score", "BLEU-4"]'
    assert extract_eval_metrics(pdf, model="ollama/test") == ["F1 score", "BLEU-4"]
    call = fake_litellm.calls[0]
    assert call["model"] == "ollama/test"
    sent = call["messages"][1]["content"]
    assert "Table 1: F1 and BLEU" in sent and "Our model improves F1" in sent
    assert "We propose a model" not in sent and "METEOR" not in sent and "ROUGE" not in sent


def test_model_from_env(pdf, fake_litellm, monkeypatch):
    monkeypatch.setenv("EVAL_METRICS_MODEL", "gpt-test")
    fake_litellm.reply = '["Accuracy"]'
    assert extract_eval_metrics(pdf) == ["Accuracy"]
    assert fake_litellm.calls[0]["model"] == "gpt-test"


def test_llm_failure_falls_back(pdf, fake_litellm):
    fake_litellm.error = RuntimeError("connection refused")
    with pytest.warns(UserWarning, match="connection refused"):
        assert extract_eval_metrics(pdf, top_k=2, model="ollama/test") == ["F1", "BLEU"]


def test_no_experiments_section_falls_back(tmp_path, fake_litellm):
    p = tmp_path / "noexp.pdf"
    SimpleDocTemplate(str(p)).build([Paragraph("We report accuracy.", getSampleStyleSheet()["BodyText"])])
    with pytest.warns(UserWarning, match="no Experiments section"):
        assert extract_eval_metrics(str(p), model="ollama/test") == ["Accuracy"]
    assert fake_litellm.calls == []


def test_cli_llm_and_show_section(pdf, fake_litellm, capsys):
    from eval_metrics.__main__ import main
    fake_litellm.reply = '["F1", "BLEU"]'
    assert main([pdf, "--model", "ollama/test", "--api-base", "http://host:11434"]) == 0
    assert capsys.readouterr().out.split() == ["F1", "BLEU"]
    assert fake_litellm.calls[0]["api_base"] == "http://host:11434"
    assert main([pdf, "--show-section"]) == 0
    out = capsys.readouterr().out
    assert "Our model improves F1" in out and "Abstract" not in out
