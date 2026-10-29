"""Static integration-contract tests for the browser client and local API.

These tests verify wiring, not a live browser session or a deployed model.
"""
from pathlib import Path

APP_JS = Path(__file__).resolve().parent.parent / "assets" / "js" / "quantai.js"
SOURCE = APP_JS.read_text(encoding="utf-8")


def test_local_inference_is_opt_in_and_targets_chat_endpoint():
    assert "window.QUANTAI_LOCAL_AI_URL" in SOURCE
    assert 'LOCAL_AI_URL + "/chat"' in SOURCE
    assert 'return LOCAL_AI_URL ? JSON.stringify({ reply: data.reply, action: data.action ?? null })' in SOURCE


def test_local_api_request_uses_expected_fields():
    assert "message: String(last.content || mode).slice(0, 9500)" in SOURCE
    assert 'context: String(context || "").slice(0, 9000)' in SOURCE
    assert "history: messages.slice(0, -1).slice(-8)" in SOURCE


def test_regression_offset_is_passed_to_deterministic_engine():
    assert 'offset: a.offset ? need(a.offset, "the exposure offset", ["continuous", "count"]) : ""' in SOURCE
    assert "offset: p.offset ? V(p.offset) : null" in SOURCE
    assert "const offsetLog = offset ? rows.map" in SOURCE
    assert 'offsetLog: model === "poisson" ? offsetLog : undefined' in SOURCE
    assert 'if (offset && opts.model !== "poisson")' in SOURCE


def test_offset_is_included_in_exported_poisson_code():
    assert "exposure(${offset.name})" in SOURCE
    assert "offset(log(${offset.name}))" in SOURCE
    assert "offset=np.log(" in SOURCE
