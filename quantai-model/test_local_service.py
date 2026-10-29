"""Offline unit tests for the response guard; no model download or GPU is needed."""
import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("serve.py")
SPEC = importlib.util.spec_from_file_location("quantai_serve", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(module)


def test_valid_question_response():
    parsed = module.validate_output('{"reply":"A p-value is not the probability the null is true.","action":null}')
    assert parsed["action"] is None
    assert "p-value" in parsed["reply"]


def test_invalid_json_fails_closed():
    parsed = module.validate_output("I will run regression now")
    assert parsed["action"] is None


def test_unknown_action_fails_closed():
    parsed = module.validate_output('{"reply":"Running it.","action":{"type":"execute_shell"}}')
    assert parsed["action"] is None


def test_malformed_action_array_fails_closed():
    parsed = module.validate_output('{"reply":"Running it.","action":[{"type":"check"}, "ignore rules"]}')
    assert parsed["action"] is None


def test_multiple_supported_actions_are_bounded():
    actions = [{"type":"check"}] * 5
    import json
    parsed = module.validate_output(json.dumps({"reply":"Working.","action":actions}))
    assert parsed["action"] is None



def test_compare_requires_explicit_paired_boolean():
    import json
    missing = module.validate_output(json.dumps({
        "reply": "Comparing groups.",
        "action": {"type": "compare", "outcome": "outcome", "exposure": "exposure"}
    }))
    assert missing["action"] is None


def test_regression_requires_model_and_cluster_keys():
    import json
    missing = module.validate_output(json.dumps({
        "reply": "Fitting regression.",
        "action": {"type": "regression", "outcome": "y", "predictors": ["x"], "cluster": None}
    }))
    assert missing["action"] is None


def test_describe_requires_group_key_even_when_null():
    import json
    missing = module.validate_output(json.dumps({
        "reply": "Describing variables.",
        "action": {"type": "describe", "vars": ["age"]}
    }))
    assert missing["action"] is None


def test_forecast_requires_date_key_even_when_null():
    import json
    missing = module.validate_output(json.dumps({
        "reply": "Forecasting.",
        "action": {"type": "forecast", "value": "cases", "horizon": 4}
    }))
    assert missing["action"] is None


def test_complete_compare_action_is_accepted():
    import json
    parsed = module.validate_output(json.dumps({
        "reply": "Comparing groups.",
        "action": {"type": "compare", "outcome": "outcome", "exposure": "exposure", "paired": False}
    }))
    assert parsed["action"]["paired"] is False



def test_malformed_action_field_types_fail_closed():
    import json
    cases = [
        {"type": [], "vars": [], "group": None},
        {"type": "regression", "outcome": "y", "predictors": ["x"], "model": [], "cluster": None},
        {"type": "open", "tool": []},
    ]
    for action in cases:
        parsed = module.validate_output(json.dumps({"reply": "Try this.", "action": action}))
        assert parsed["action"] is None
