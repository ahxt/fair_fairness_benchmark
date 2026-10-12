"""Weighted crawling of metric/value evidence across academic papers.

Scores are transparent heuristics, not calibrated probabilities. Repeated
independent mentions provide statistical support; no training data is required.
"""

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from team.eval_metrics_tool.eval_metrics.extraction import PaperContent, extract_paper
from team.eval_metrics_tool.eval_metrics.matcher import find_metrics, normalize_metric
from team.eval_metrics_tool.eval_metrics.scoring import rank_metrics
from team.eval_metrics_tool.eval_metrics.sections import split_sections

MetricValue = Optional[Union[int, float]]

# Each feature's contribution can be overridden through the initializer.
VALUE_WEIGHTS = {
    'proximity': 4.0,          # exponentially decays with character distance
    'direct_statement': 3.0,  # e.g. "F1 = 0.87" or "92% accuracy"
    'result_cue': 3.0,        # achieved, obtained, reached, reported, ...
    'our_method': 2.0,        # our/proposed method; table row labeled Ours
    'table_alignment': 5.0,   # value aligned with a metric header/row label
    'results_section': 1.0,   # optional hint; unrecognized sections score normally
    'summary_section': 0.5,
    'baseline': -3.0,
    'related_work': -3.0,
    'references': -6.0,
    'citation': -6.0,
    'year_or_count': -5.0,
    'delta': -5.0,            # improvement amount rather than absolute result
    'repeat_support': 2.0,    # logarithmic bonus for independent repetitions
}
_NUMBER = r'[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?'
_NUMBERS = re.compile(rf'(?<![\w.])(?P<value>{_NUMBER})(?!\w|\.\d)\s*(?P<percent>%)?')
_CELL = re.compile(rf'^\s*({_NUMBER})\s*%?(?:\s*(?:±|\+/-)\s*{_NUMBER}\s*%?)?\s*[*†‡]*\s*$')
_DIRECT = re.compile(r'^\s*(?:(?:score|value)\s*)?(?:[:=]|(?:of|is|was|at)\b)?\s*$', re.I)
_RESULT = re.compile(r'\b(?:achiev\w*|obtain\w*|reach\w*|report\w*|yield\w*|scor\w*|measur\w*|evaluat\w*|record\w*|results?\s+(?:show|indicate|demonstrate))\b', re.I)
_OURS = re.compile(r'\b(?:ours|our(?:\s+\w+){0,3}|proposed(?:\s+(?:method|model|approach))?)\b', re.I)
_BASELINE = re.compile(r'\b(?:baseline|previous|prior|existing|competing|competitor)\b', re.I)
_COUNT = re.compile(r'^\s*(?:samples?|examples?|epochs?|parameters?|participants?|datasets?|runs?|iterations?)\b', re.I)
_DELTA = re.compile(r'\b(?:improv\w*|increas\w*|decreas\w*|gain\w*|reduc\w*)\s+(?:\w+\s+){0,3}(?:by|of)\s*$', re.I)
# Sentence punctuation, excluding decimal points. Do not require PDF headings.
_SENTENCE_BREAK = re.compile(r'(?<=[!?;])\s+|(?<!\d)\.\s+|(?<=\d)\.(?!\d)\s+')
_TABLE_VALUE = re.compile(
    rf'(?<![\w.])(?P<value>{_NUMBER})(?:\s*(?:±|\+/-)\s*{_NUMBER})?\s*%?(?!\w)'
)
# Common compact headers in ML result tables, including FFB Table 4. The
# teammate's matcher deliberately treats short acronyms more conservatively in
# prose, so table header aliases are handled here without changing that rule.
_TABLE_HEADERS = {
    'acc': 'Accuracy', 'auc': 'AUC', 'ap': 'mAP', 'f1': 'F1',
    'dp': 'Demographic Parity', 'abcc': 'ABCC', 'prule': 'p%-rule',
    'eodd': 'Equalized Odds', 'eopp': 'Equal Opportunity',
}


@dataclass(frozen=True)
class _Evidence:
    metric: str
    value: float
    score: float
    context: str
    features: tuple[str, ...]


def _section_features(kind):
    return {
        'results': ['results_section'],
        'abstract': ['summary_section'],
        'conclusion': ['summary_section'],
        'related_work': ['related_work'],
        'references': ['references'],
    }.get(kind, [])


def _context_features(text):
    features = []
    if _RESULT.search(text):
        features.append('result_cue')
    if _OURS.search(text):
        features.append('our_method')
    if _BASELINE.search(text):
        features.append('baseline')
    return features


def _crawl_text(paper, wanted, weights, window):
    evidence = []
    for section in paper.sections:
        for sentence in _SENTENCE_BREAK.split(section.text):
            # Joining line wraps lets indirect wording span PDF lines.
            text = ' '.join(sentence.split())
            mentions = find_metrics(text, context_given=True)
            numbers = [n for n in _NUMBERS.finditer(text)
                       if not any(n.start() < m.end and n.end() > m.start
                                  for m in mentions)]
            for index, number in enumerate(numbers):
                # ± values are uncertainty, not independent metric results.
                if re.search(r'(?:±|\+/-)\s*$', text[:number.start()]):
                    continue
                distances = [(max(m.start - number.end(), number.start() - m.end, 0), m)
                             for m in mentions]
                if not distances:
                    continue
                distance, mention = min(distances, key=lambda pair: pair[0])
                if distance > window or mention.metric not in wanted:
                    continue
                after = number.start() >= mention.end
                bridge = (text[mention.end:number.start()] if after
                          else text[number.end():mention.start])
                # Local cues stop at the previous/next value, so baseline cues
                # do not automatically contaminate another model's result.
                lo = numbers[index - 1].end() if index else max(0, number.start() - window)
                hi = numbers[index + 1].start() if index + 1 < len(numbers) else min(len(text), number.end() + window)
                local = text[lo:number.end()] if after else text[number.start():hi]
                if after and mention.start >= lo:
                    local = text[lo:number.end()]
                features = _section_features(section.kind) + _context_features(local)
                direct = bool(_DIRECT.fullmatch(bridge)) if after else not bridge.strip()
                if direct:
                    features.append('direct_statement')
                value = float(number.group('value'))
                if not math.isfinite(value):
                    continue
                prefix = text[max(0, number.start() - 100):number.start()]
                suffix = text[number.end():]
                if prefix.count('[') > prefix.count(']'):
                    features.append('citation')
                if (1900 <= value <= 2100 and value.is_integer()
                        or _COUNT.match(suffix)):
                    features.append('year_or_count')
                if _DELTA.search(prefix) or re.match(r'\s*(?:percentage\s+points?|points?)\b', suffix, re.I):
                    features.append('delta')
                proximity = weights['proximity'] * math.exp(-distance / 40.0)
                score = proximity + sum(weights[f] for f in features)
                evidence.append(_Evidence(mention.metric, value, score, text,
                                          tuple(['proximity'] + features)))
    return evidence


def _crawl_tables(paper, wanted, weights, row_label):
    evidence = []
    found = False
    for table in paper.tables:
        headers = {}
        for row in table.rows:
            labels = {}
            values = {}
            for col, cell in enumerate(row):
                matches = find_metrics(cell, context_given=True)
                if len(matches) == 1:
                    labels[col] = matches[0].metric
                number = _CELL.fullmatch(cell)
                if number and math.isfinite(value := float(number.group(1))):
                    values[col] = value
            selected = row_label is None or any(
                cell.strip().casefold() == row_label.strip().casefold() for cell in row)
            found |= selected and row_label is not None
            if selected:
                context = ' | '.join(row)
                features = ['table_alignment'] + _section_features(table.section) + _context_features(context)
                score = sum(weights[f] for f in features)
                pairs = [(headers[col], value) for col, value in values.items() if col in headers]
                if len(labels) == 1 and len(values) == 1:
                    label_col, metric = next(iter(labels.items()))
                    value_col, value = next(iter(values.items()))
                    if value_col > label_col:
                        pairs.append((metric, value))
                for metric, value in pairs:
                    if metric in wanted:
                        evidence.append(_Evidence(metric, value, score, context, tuple(features)))
            if labels and not values:
                # A repeated header starts a new block, not a stale merge.
                headers = labels
    return evidence, found


def _crawl_unruled_table_row(paper, wanted, weights, row_label):
    """Read a selected row when PDF extraction leaves a table as plain text.

    A preceding whitespace-separated metric header supplies the column order.
    The caller must specify the row label: guessing a row from a paper with
    several datasets or methods would produce an arbitrary baseline.
    """
    evidence = []
    label = re.compile(
        r'^\s*' + r'\s+'.join(map(re.escape, row_label.split())) + r'\s+(?=[-+]?\d)',
        re.I,
    )
    for section in paper.sections:
        if section.kind == 'references':
            continue
        headers = []
        header_line = -100
        for line_number, line in enumerate(section.text.splitlines()):
            tokens = line.strip().split()
            recognized = [(_TABLE_HEADERS.get(token.lower()) or normalize_metric(token))
                          for token in tokens]
            recognized = [name for name in recognized if name]
            if recognized and len(recognized) >= len(tokens) - 2:
                headers = recognized
                header_line = line_number
                continue
            match = label.match(line)
            if not match or not headers or line_number - header_line > 40:
                continue
            values = [float(m.group('value')) for m in _TABLE_VALUE.finditer(line[match.end():])]
            if len(values) < len(headers):
                continue
            context = line.strip()
            features = ['table_alignment'] + _section_features(section.kind)
            score = sum(weights[f] for f in features)
            for metric, value in zip(headers, values):
                if metric in wanted and math.isfinite(value):
                    evidence.append(_Evidence(metric, value, score, context,
                                              tuple(features)))
    return evidence


def _rank_values(evidence, weights):
    """Aggregate support without letting duplicated snippets dominate."""
    grouped = defaultdict(dict)
    for item in evidence:
        if item.score <= 0:
            continue
        contexts = grouped[(item.metric, item.value)]
        contexts[item.context] = max(contexts.get(item.context, 0), item.score)
    ranked = defaultdict(list)
    for (metric, value), contexts in grouped.items():
        scores = sorted(contexts.values(), reverse=True)
        # Repetition contributes sublinearly and is discounted by evidence strength.
        support = sum(min(score / max(scores[0], 1), 1) for score in scores[1:])
        score = scores[0] + weights['repeat_support'] * math.log1p(support)
        ranked[metric].append((value, score))
    for values in ranked.values():
        values.sort(key=lambda pair: (-pair[1], pair[0]))
    return ranked


def create_eval_metrics_record(
    pdf_path: str,
    metrics: Optional[Iterable[str]] = None,
    *,
    top_k: int = 5,
    row_label: Optional[str] = None,
    weights: Optional[Mapping[str, float]] = None,
    window: int = 140,
    min_score: float = 4.0,
    min_margin: float = 1.0,
) -> dict[str, MetricValue]:
    """Crawl a PDF or UTF-8 .txt paper and initialize weighted metric values.

    Use supplied metric names (including extract_eval_metrics' output), or
    the existing top_k metric-name ranking. Score nearby numeric candidates
    using proximity, linguistic cues, table alignment, optional section hints
    and logarithmic repetition support. All sections are crawled; headings
    and exact metric/value wording are not required.

    Return None when no candidate meets min_score or the top two candidates
    are separated by less than min_margin. This is heuristic ranking, not a
    calibrated confidence probability or a guarantee of the intended model.
    Optional row_label restricts evidence to exact matching table rows.
    weights overrides entries in VALUE_WEIGHTS. Percentages retain their
    printed scale; the central estimate of an uncertainty interval is used.
    """
    if metrics is None and top_k < 1:
        raise ValueError('top_k must be >= 1')
    if isinstance(metrics, str):
        raise TypeError('metrics must be a collection of names, not a string')
    if window < 1 or not math.isfinite(min_score) or not math.isfinite(min_margin) or min_margin < 0:
        raise ValueError('window must be positive and thresholds must be finite; min_margin must be nonnegative')
    if row_label is not None and not row_label.strip():
        raise ValueError('row_label must not be empty')
    scoring = dict(VALUE_WEIGHTS)
    if weights is not None:
        unknown = set(weights) - scoring.keys()
        if unknown:
            raise ValueError(f'Unknown weight features: {sorted(unknown)}')
        if any(not math.isfinite(v) for v in weights.values()):
            raise ValueError('weights must be finite')
        scoring.update(weights)
    path = Path(pdf_path)
    paper = (PaperContent(split_sections(path.read_text(encoding='utf-8')))
             if path.suffix.lower() == '.txt' else extract_paper(str(path)))
    names = ([score.name for score in rank_metrics(paper)[:top_k]]
             if metrics is None else [normalize_metric(name) or name for name in metrics])
    record = dict.fromkeys(names, None)
    evidence, found_row = _crawl_tables(paper, record, scoring, row_label)
    if row_label is None:
        evidence.extend(_crawl_text(paper, record, scoring, window))
    elif not evidence:
        evidence.extend(_crawl_unruled_table_row(paper, record, scoring, row_label))
        if not found_row and not evidence:
            raise ValueError(f'No table row labeled {row_label!r} was found')
    ranked = _rank_values(evidence, scoring)
    for metric in record:
        values = ranked.get(metric, [])
        if values and values[0][1] >= min_score:
            if len(values) == 1 or values[0][1] - values[1][1] >= min_margin:
                record[metric] = values[0][0]
    return record


def insert_eval_metric(
    record: dict[str, MetricValue], metric: str, value: MetricValue
) -> dict[str, MetricValue]:
    """Update an initialized metric in place and return the same record."""
    if metric not in record:
        raise KeyError(f'Metric {metric!r} is not initialized in the record')
    record[metric] = value
    return record
