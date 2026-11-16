"""Render a portable, evidence-backed ETF case report without network or models."""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
from html import escape
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from urllib.parse import urlsplit


_LABELS = {"equal_weight": "Daily equal weight", "fixed_momentum": "Fixed momentum",
           "selected_grid": "Selected grid candidate"}
_COLORS = ("#64748b", "#0891b2", "#c05621")
_METRICS = ("total_return", "annualized_return", "sharpe", "max_drawdown", "mean_turnover")


def _number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _reject_constant(value: str):
    raise ValueError(f"Non-finite JSON constant: {value}")


def _text(value) -> str:
    return "Unknown" if value is None else str(value)


def _pct(value) -> str:
    return "Unknown" if value is None else f"{_number(value, 'percentage'):+.2%}"


def _pp(value) -> str:
    return "Unknown" if value is None else f"{_number(value, 'excess return') * 100:+.2f} pp"


def _metric(value, field: str) -> str:
    if value is None:
        return "Undefined" if field == "sharpe" else "Unknown"
    amount = _number(value, field)
    return f"{amount:.2f}" if field == "sharpe" else f"{amount:.2%}"


def _label(item: dict) -> str:
    label = _LABELS.get(item["id"], item["id"])
    candidate = item.get("candidate")
    if isinstance(candidate, dict):
        label += f" ({candidate.get('id', 'Unknown')})"
    return label


def _table(headers: list[str], rows: list[list[str]]) -> str:
    return '<div class="table-scroll"><table><thead><tr>' + ''.join(
        f"<th>{escape(value)}</th>" for value in headers
    ) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(
        f"<td>{escape(value)}</td>" for value in row
    ) + '</tr>' for row in rows) + '</tbody></table></div>'


def _md_text(value: str) -> str:
    text = escape(value).replace("\\", "\\\\").replace("\n", " ")
    for character in "[]`*_":
        text = text.replace(character, "\\" + character)
    return text


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    def cell(value: str) -> str:
        return _md_text(value).replace("|", "&#124;")
    return '\n'.join('| ' + ' | '.join(cell(value) for value in row) + ' |'
                     for row in [headers, ['---'] * len(headers), *rows])


def _source_html(value) -> str:
    text = _text(value)
    try:
        parsed = urlsplit(text)
    except ValueError:
        return escape(text)
    if parsed.scheme in ("http", "https") and parsed.netloc and not any(c.isspace() for c in text):
        return f'<a href="{escape(text, quote=True)}" rel="noreferrer">{escape(text)}</a>'
    return escape(text)


def _drawdowns(equity: list[float]) -> list[float]:
    peak = equity[0]
    values = []
    for value in equity:
        peak = max(peak, value)
        values.append(value / peak - 1.0)
    return values


def _validate_result(result: dict) -> None:
    if not isinstance(result, dict) or result.get("schema") != "quant_backtest/1":
        raise ValueError("Expected quant_backtest/1 result")
    dates = result.get("return_dates")
    if not isinstance(dates, list) or not dates:
        raise ValueError("Backtest needs observed return dates")
    try:
        parsed = [date.fromisoformat(value) for value in dates]
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid backtest dates") from exc
    if parsed != sorted(set(parsed)) or [day.isoformat() for day in parsed] != dates:
        raise ValueError("Backtest dates must be unique and increasing")
    count = len(dates)
    for field, size in (("net_returns", count), ("turnovers", count), ("equity", count + 1)):
        values = result.get(field)
        if not isinstance(values, list) or len(values) != size:
            raise ValueError(f"Backtest {field} length does not match its dates")
        for value in values:
            _number(value, field)
    equity = result["equity"]
    if any(value <= 0 for value in equity) or not math.isclose(equity[0], 1.0, abs_tol=1e-12):
        raise ValueError("Backtest equity must start at one and remain positive")
    if any(value < 0 for value in result["turnovers"]):
        raise ValueError("Backtest turnover cannot be negative")
    for before, after, net in zip(equity, equity[1:], result["net_returns"]):
        if not math.isclose(after, before * (1 + net), rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Backtest equity disagrees with daily net returns")
    metrics = result.get("metrics", {})
    for name in _METRICS:
        if name not in metrics or (name != "sharpe" or metrics[name] is not None):
            _number(metrics.get(name), name)
    expected = {"total_return": equity[-1] - 1.0,
                "annualized_return": equity[-1] ** (252 / count) - 1.0,
                "max_drawdown": -min(_drawdowns(equity)),
                "mean_turnover": fmean(result["turnovers"])}
    for name, value in expected.items():
        if not math.isclose(metrics[name], value, rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError(f"Backtest {name} disagrees with daily evidence")
    volatility = pstdev(result["net_returns"])
    expected_sharpe = fmean(result["net_returns"]) / volatility * math.sqrt(252) if volatility else None
    if (expected_sharpe is None) != (metrics["sharpe"] is None) or (expected_sharpe is not None and not
            math.isclose(metrics["sharpe"], expected_sharpe, rel_tol=1e-8, abs_tol=1e-10)):
        raise ValueError("Backtest Sharpe disagrees with daily evidence")


def _validate_window(window: dict, result: dict) -> None:
    dates = result["return_dates"]
    if window.get("start") != dates[0] or window.get("end") != dates[-1] or window.get("observations") != len(dates):
        raise ValueError("Reported window differs from observed return dates")


def _validate_comparison(series: list[dict]) -> None:
    calendar = None
    reference = None
    for item in series:
        _validate_result(item.get("result"))
        result = item["result"]
        if calendar is not None and calendar != result["return_dates"]:
            raise ValueError("Comparison series have different calendars")
        calendar = result["return_dates"]
        if item["id"] == "equal_weight":
            reference = result["metrics"]["total_return"]
    if reference is not None:
        for item in series:
            expected = item["result"]["metrics"]["total_return"] - reference
            if not math.isclose(_number(item.get("excess_return"), "excess_return"), expected,
                                rel_tol=1e-8, abs_tol=1e-10):
                raise ValueError("Excess return disagrees with equal-weight baseline")


def _chart(series: list[dict], *, drawdown: bool, title: str) -> str:
    """Plot every observed equity point; initial capital is not assigned a date."""
    curves = [_drawdowns(item["result"]["equity"]) if drawdown else item["result"]["equity"]
              for item in series]
    low = min(min(values) for values in curves)
    high = max(max(values) for values in curves)
    span = high - low or max(abs(high) * .05, .01)
    low, high = low - span * .07, high + span * .07
    width, height = 1000, 365
    left, top, right, bottom = 77, 54, 25, 55
    plot_w, plot_h = width - left - right, height - top - bottom
    count = len(curves[0])

    def x(index):
        return left + index / (count - 1) * plot_w

    def y(value):
        return top + (high - value) / (high - low) * plot_h

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
             f'role="img" aria-label="{escape(title, quote=True)}">',
             f'<title>{escape(title)}</title>',
             f'<desc>Every daily observation is plotted. {count - 1} return dates per series. '
             'Initial capital is one before the first return.</desc>',
             '<rect width="100%" height="100%" fill="#fff"/>',
             f'<text x="{left}" y="27" font-size="16" fill="#172a3a">{escape(title)}</text>']
    for index in range(5):
        value = low + (high - low) * index / 4
        label = f"{value:.1%}" if drawdown else f"{value:.2f}"
        parts += [f'<line x1="{left}" y1="{y(value):.2f}" x2="{width-right}" '
                  f'y2="{y(value):.2f}" stroke="#e2e8f0"/>',
                  f'<text x="{left-10}" y="{y(value)+4:.2f}" text-anchor="end" '
                  f'font-size="12" fill="#526172">{label}</text>']
    dates = series[0]["result"]["return_dates"]
    ticks = sorted(set((0, 1, count // 2, count - 1)))
    # Start and first-return labels would overlap on long daily histories.
    if count > 30:
        ticks.remove(1)
    for index in ticks:
        label = "Initial" if index == 0 else dates[index - 1]
        anchor = "start" if index == 0 else ("end" if index == count - 1 else "middle")
        parts.append(f'<text x="{x(index):.2f}" y="{height-20}" text-anchor="{anchor}" '
                     f'font-size="12" fill="#526172">{label}</text>')
    for index, (item, values) in enumerate(zip(series, curves)):
        points = ' '.join(f"{x(point):.2f},{y(value):.2f}" for point, value in enumerate(values))
        parts.append(f'<polyline data-series="{escape(item["id"], quote=True)}" '
                     f'data-observations="{len(values)}" fill="none" stroke="{_COLORS[index % 3]}" '
                     f'stroke-width="2" points="{points}"/>')
    parts.append('</svg>')
    return ''.join(parts)


def _metric_rows(series: list[dict]) -> list[list[str]]:
    return [[_label(item), *(_metric(item["result"]["metrics"][field], field) for field in _METRICS),
             _pp(item.get("excess_return"))] for item in series]


def generate_report(study_path: Path, output_dir: Path) -> Path:
    """Build index.html, report.md, SVGs and an exact study.json evidence copy.

    The supplied study is the trust boundary: no market download, model call,
    rerun or substitute result is attempted here. Invalid daily evidence raises.
    """
    raw = Path(study_path).read_bytes()
    study = json.loads(raw, parse_constant=_reject_constant)
    if not isinstance(study, dict) or study.get("schema") != "etf_case_study/1":
        raise ValueError("Expected etf_case_study/1 study")
    protocol = study.get("protocol", {})
    windows = study.get("windows", {})
    development = study.get("development", {})
    candidates = development.get("candidates", [])
    baselines = development.get("baselines", [])
    selected = study.get("selected_candidate") or {}
    selected_id = selected.get("id")
    dev_windows = windows.get("development", [])
    window_ids = [window["id"] for window in dev_windows]
    if len(set(window_ids)) != len(window_ids):
        raise ValueError("Development window IDs must be unique")
    for item in candidates + baselines:
        evidence_ids = [window["window_id"] for window in item.get("windows", [])]
        if len(set(evidence_ids)) != len(evidence_ids) or any(name not in window_ids for name in evidence_ids):
            raise ValueError("Candidate has duplicate or unknown development windows")
        for window in item.get("windows", []):
            _validate_result(window.get("result"))
            _validate_window(next(value for value in dev_windows if value["id"] == window["window_id"]), window["result"])
            _number(window.get("excess_return"), "development excess return")
        excesses = [window["excess_return"] for window in item.get("windows", [])]
        if item in candidates and excesses:
            if not math.isclose(_number(item.get("worst_excess_return"), "worst excess"), min(excesses),
                                rel_tol=1e-8, abs_tol=1e-10) or not math.isclose(
                    _number(item.get("mean_excess_return"), "mean excess"), fmean(excesses),
                    rel_tol=1e-8, abs_tol=1e-10):
                raise ValueError("Candidate development summary disagrees with its windows")
    for window_id in window_ids:
        comparison = [dict(id=item["id"], **next(
            window for window in item["windows"] if window["window_id"] == window_id))
            for item in baselines + candidates
            if any(window["window_id"] == window_id for window in item.get("windows", []))]
        _validate_comparison(comparison)

    holdout = study.get("holdout") or {}
    series = holdout.get("series") or []
    holdout_ready = study.get("status") == "completed" and holdout.get("status") == "completed" \
        and len(series) == 3 and {item.get("id") for item in series} == set(_LABELS)
    if holdout_ready:
        series = sorted(series, key=lambda item: list(_LABELS).index(item["id"]))
        _validate_comparison(series)
        for item in series:
            _validate_window(windows.get("holdout", {}), item["result"])
    digest = hashlib.sha256(raw).hexdigest()
    title = "ETF research case · fixed universe, stated costs"
    qwen = study.get("qwen") or {}
    qwen_status = "Qwen NOT RUN" if qwen.get("status") == "not_run" else f"Qwen status: {_text(qwen.get('status'))}"
    notice = f"{qwen_status}. This is a deterministic grid study; its results are not Agent performance."
    html = [f'<header><p class="eyebrow">REPRODUCIBLE RESEARCH CASE</p><h1>{title}</h1>'
            f'<p class="notice">{escape(notice)}</p><p>Study status: <strong>{escape(_text(study.get("status")))}</strong>'
            f' · Protocol: {escape(_text(protocol.get("id")))}</p></header>']
    md = [f"# {title}", f"**{_md_text(notice)}**", f"Study status: {_md_text(_text(study.get('status')))}."]

    def section(heading: str, body: str, markdown: str):
        html.append(f'<section><h2>{escape(heading)}</h2>{body}</section>')
        md.extend((f"## {heading}", markdown))

    question = _text(protocol.get("question"))
    universe = ', '.join(str(value) for value in protocol.get("symbols", [])) or "Unknown"
    window_rows = [["Historical context only", _text(study.get("provenance", {}).get("actual_start_date") or protocol.get("data_start")),
                    _text(windows.get("warmup_end")), "No model fitting"]]
    for window in dev_windows + ([windows["holdout"]] if windows.get("holdout") else []):
        window_rows.append([_text(window.get("id")), _text(window.get("start")),
                            _text(window.get("end")), _text(window.get("observations"))])
    headers = ["Window", "Start", "End", "Return observations / role"]
    section("1 · Research question and frozen design",
            f'<p>{escape(question)}</p><p><strong>Fixed universe:</strong> {escape(universe)}.</p>'
            + _table(headers, window_rows),
            f"{_md_text(question)}\n\nFixed universe: {_md_text(universe)}.\n\n" + _markdown_table(headers, window_rows))

    selection_notes = _text(protocol.get("selection_notes") or development.get("selection_rule"))
    dev_headers = ["Candidate", *window_ids, "Worst excess", "Mean excess", "Outcome"]
    dev_rows = []
    for item in candidates:
        values = {window["window_id"]: window["excess_return"] for window in item.get("windows", [])}
        outcome = "Selected on development" if item["id"] == selected_id else "Rejected by frozen rule"
        if values and all(value < 0 for value in values.values()):
            outcome += "; below equal weight in every observed dev window"
        elif values and min(values.values()) < 0:
            outcome += "; negative dev window retained"
        dev_rows.append([item["id"], *(_pp(values.get(name)) for name in window_ids),
                         _pp(item.get("worst_excess_return")), _pp(item.get("mean_excess_return")), outcome])
    dev_copy = (f"Selection: {_text(selected_id)}. {selection_notes} "
                "Excess means the difference in total net return versus daily equal weight in the same window, "
                "in percentage points; it is not annualized alpha. Negative and rejected outcomes are retained.")
    section("2 · Development selection, including negative results",
            f'<p>{escape(dev_copy)}</p>' + (_table(dev_headers, dev_rows) if dev_rows else '<p>No candidate results available.</p>'),
            _md_text(dev_copy) + '\n\n' + (_markdown_table(dev_headers, dev_rows) if dev_rows else "No candidate results available."))

    metric_headers = ["Strategy", "Net return", "Annualized", "Sharpe", "Max drawdown", "Mean daily turnover", "Net excess vs EW"]
    charts = {}

    def figures(comparison: list[dict], prefix: str, heading: str) -> tuple[str, str]:
        legend = ''.join(f'<li><span style="background:{_COLORS[index % 3]}"></span>{escape(_label(item))}</li>'
                         for index, item in enumerate(comparison))
        body = f'<ul class="legend">{legend}</ul>'
        markdown = []
        for drawdown, name in ((False, "net-equity"), (True, "drawdown")):
            filename = f"{prefix}-{name}.svg"
            charts[filename] = _chart(comparison, drawdown=drawdown,
                                      title=f"{heading} · {'drawdown from peak' if drawdown else 'net equity (initial = 1)'}")
            body += f'<figure>{charts[filename]}</figure>'
            markdown.append(f"![{_md_text(heading)} {name}]({filename})")
        return body, '\n\n'.join(markdown)

    if holdout_ready:
        chart_html, chart_md = figures(series, "holdout", "Holdout")
        chosen = next(item for item in series if item["id"] == "selected_grid")
        holdout_copy = (f"The development-selected candidate ({_text(selected_id)}) has holdout net excess "
                        f"{_pp(chosen['excess_return'])} versus daily equal weight. "
                        "This is one historical holdout evaluation, with no retuning after selection.")
        section("3 · Holdout comparison", f'<p>{escape(holdout_copy)}</p>'
                + _table(metric_headers, _metric_rows(series)) + chart_html,
                _md_text(holdout_copy) + '\n\n' + _markdown_table(metric_headers, _metric_rows(series)) + '\n\n' + chart_md)
    else:
        reason = (study.get("error") or {}).get("message")
        failure = (f"NO COMPLETE HOLDOUT RESULT. Holdout status: {_text(holdout.get('status'))}. "
                   "Three completed comparison series are required. No performance curve is shown."
                   + (f" Recorded error: {reason}" if reason else ""))
        section("3 · Holdout comparison", f'<p class="notice">{escape(failure)}</p>', f"**{_md_text(failure)}**")

    dev_details, dev_md = [], []
    for index, window in enumerate(dev_windows):
        comparison = []
        for name, source in (("equal_weight", next((item for item in baselines if item["id"] == "equal_weight"), None)),
                             ("fixed_momentum", next((item for item in baselines if item["id"] == "fixed_momentum"), None)),
                             ("selected_grid", next((item for item in candidates if item["id"] == selected_id), None))):
            evidence = next((value for value in (source or {}).get("windows", []) if value["window_id"] == window["id"]), None)
            if evidence:
                comparison.append({**evidence, "id": name, "candidate": selected if name == "selected_grid" else (source or {}).get("candidate")})
        if comparison:
            chart_html, chart_md = figures(comparison, f"development-{index + 1}", window["id"])
            dev_details.append(f'<details><summary>{escape(window["id"])}</summary>'
                               + _table(metric_headers, _metric_rows(comparison)) + chart_html + '</details>')
            dev_md.append(f"### {_md_text(window['id'])}\n\n" + _markdown_table(metric_headers, _metric_rows(comparison)) + '\n\n' + chart_md)
    if dev_details:
        section("4 · Development evidence", ''.join(dev_details), '\n\n'.join(dev_md))

    assumption_rows = [["Fee (bps per traded notional)", _text(protocol.get("fee_bps"))],
                       ["Slippage (bps per traded notional)", _text(protocol.get("slippage_bps"))],
                       ["Selected positions", _text(protocol.get("top_k"))],
                       ["Fixed momentum lookback (sessions)", _text(protocol.get("fixed_momentum_lookback"))]]
    execution = _text(protocol.get("execution_notes"))
    definitions = ("Daily net equity compounds the recorded net returns. Drawdown is equity / running peak − 1. "
                   "Maximum drawdown is reported as a positive loss magnitude. Annualized return and Sharpe use 252 sessions/year; "
                   "Sharpe uses zero risk-free return and population daily volatility. Turnover is gross traded notional "
                   "(sum of absolute weight changes); each window starts from cash. Undefined Sharpe remains undefined.")
    section("5 · Execution and metric definitions", _table(["Assumption", "Value"], assumption_rows)
            + f'<p>{escape(execution)}</p><p>{escape(definitions)}</p>',
            _markdown_table(["Assumption", "Value"], assumption_rows) + f'\n\n{_md_text(execution)}\n\n{definitions}')

    provenance = study.get("provenance", {})
    source_rows = [[name.replace('_', ' ').title(), _text(provenance.get(name))]
                   for name in ("source", "license", "price_basis", "universe_policy")]
    for name in ("data_kind", "actual_start_date", "actual_end_date", "date_count", "row_count", "path_semantics"):
        if name in provenance:
            source_rows.append([name.replace('_', ' ').title(), _text(provenance[name])])
    snapshot = provenance.get("source_snapshot") or {}
    for name in ("url", "version", "last_updated", "retrieved_at_utc", "metadata_url"):
        if name in snapshot:
            source_rows.append(["Source snapshot " + name.replace('_', ' '), _text(snapshot[name])])
    missing = provenance.get("missing_date_policy") or {}
    for name in ("method", "calendar_validation", "discarded_date_count"):
        if name in missing:
            source_rows.append(["Missing dates: " + name.replace('_', ' '), _text(missing[name])])
    hashes = [["Report evidence (study.json)", digest]]
    for name in ("dataset_manifest", "dataset_csv", "protocol"):
        descriptor = provenance.get(name) or {}
        hashes.append([f"{name}: {_text(descriptor.get('path'))}", _text(descriptor.get("sha256"))])
    for descriptor in provenance.get("source_files", []):
        hashes.append([_text(descriptor.get("path")), _text(descriptor.get("sha256"))])
    if snapshot.get("metadata_sha256"):
        hashes.append(["Source metadata", _text(snapshot["metadata_sha256"])])
    for descriptor in snapshot.get("source_files", []):
        hashes.append(["Source " + _text(descriptor.get("source_file") or descriptor.get("path")), _text(descriptor.get("sha256"))])
    if study.get("dev_selection"):
        hashes.append(["Frozen dev selection", _text(study["dev_selection"].get("sha256"))])
    source_html = '<dl>' + ''.join(f'<dt>{escape(row[0])}</dt><dd>{_source_html(row[1])}</dd>' for row in source_rows) + '</dl>'
    evidence_copy = ("The report includes the exact study JSON, including daily dates, returns, weights, turnover and net equity. "
                     "Hashes identify the declared inputs; the renderer does not independently verify the upstream data license or source. "
                     "Raw market data is not embedded in this report.")
    section("6 · Data provenance and evidence", source_html + f'<p>{escape(evidence_copy)} '
            '<a href="study.json">Download study.json</a>.</p>' + _table(["Artifact", "SHA-256"], hashes),
            _markdown_table(["Field", "Declaration"], source_rows) + f'\n\n{evidence_copy}\n\n'
            '[Exact evidence: study.json](study.json)\n\n' + _markdown_table(["Artifact", "SHA-256"], hashes))

    usage = study.get("model_usage") or {}
    usage_rows = []
    for name in ("model_calls", "input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"Invalid model usage: {name}")
        usage_rows.append([name, _text(value)])
    limitations = list(study.get("limitations") or [])
    limitations.insert(0, protocol.get("universe_notes") or
                       "A fixed, surviving ETF universe cannot establish point-in-time membership or remove survivorship bias.")
    limitations.append("A Qwen comparison has not been established by this deterministic report. A future Agent study needs its own frozen protocol and an unopened final interval.")
    section("7 · Limits and model usage", f'<p><strong>{escape(notice)}</strong></p>'
            + _table(["Recorded model usage", "Value"], usage_rows)
            + '<ul>' + ''.join(f'<li>{escape(str(value))}</li>' for value in limitations) + '</ul>',
            f"**{_md_text(notice)}**\n\n" + _markdown_table(["Recorded model usage", "Value"], usage_rows)
            + '\n\n' + '\n'.join(f"- {_md_text(str(value))}" for value in limitations))

    style = """*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:#243444;font:16px/1.65 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}main{max-width:1140px;margin:auto;padding:44px 28px 70px}header,section{background:#fff;padding:28px 34px;margin-bottom:22px;border:1px solid #dde4e9;border-radius:10px}header{border-top:5px solid #c05621}h1{font-size:clamp(28px,4vw,42px);line-height:1.2;max-width:900px;margin:12px 0 24px}h2{font-size:23px;line-height:1.35;margin-top:0}p{max-width:1000px}.eyebrow{font-size:12px;letter-spacing:.16em;color:#64748b;font-weight:700}.notice{background:#fff5e8;border-left:4px solid #c05621;padding:14px 18px;font-weight:600}a{color:#076784}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:11px 12px;border-bottom:1px solid #e2e8f0;vertical-align:top}th{background:#f3f6f8;font-weight:650}td{overflow-wrap:anywhere}th:first-child,td:first-child{min-width:150px}.table-scroll{overflow-x:auto;margin:22px 0}figure{margin:22px 0}svg{width:100%;height:auto;display:block}details{border-top:1px solid #e2e8f0;padding:15px 0}summary{font-weight:650;cursor:pointer}.legend{display:flex;flex-wrap:wrap;gap:8px 24px;padding:0;list-style:none;font-size:13px}.legend span{display:inline-block;width:20px;height:3px;vertical-align:middle;margin-right:8px}dt{font-weight:650}dd{margin:3px 0 14px;overflow-wrap:anywhere}footer{font-size:13px;color:#64748b;text-align:center}@media(max-width:600px){main{padding:16px 10px}header,section{padding:22px 16px}th,td{padding:9px}svg{min-width:620px}figure{overflow-x:auto}}@media print{body{background:white}main{padding:0}header,section{break-inside:avoid;border-radius:0}details{display:block}details>summary{display:none}}"""
    document = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; img-src \'self\' data:; base-uri \'none\'">'
                f'<title>{title}</title><style>{style}</style></head><body><main>'
                + ''.join(html) + '<footer>Static report · no model calls or network requests during rendering.</footer></main></body></html>')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "study.json").write_bytes(raw)
    for filename, content in charts.items():
        (output_dir / filename).write_text(content, encoding="utf-8")
    (output_dir / "report.md").write_text('\n\n'.join(md) + '\n', encoding="utf-8")
    index_path = output_dir / "index.html"
    index_path.write_text(document, encoding="utf-8")
    return index_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(generate_report(args.study, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
