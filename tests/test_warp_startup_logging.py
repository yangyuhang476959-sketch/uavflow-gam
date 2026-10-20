"""Exercise the actual startup-log expression without loading model weights."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from torch import nn


@pytest.mark.parametrize('predictor,expected', [
    (nn.Identity(), True),
    (SimpleNamespace(variable_language_tokens=False), False),
    (SimpleNamespace(variable_language_tokens=True), True),
])
def test_startup_logging_supports_predictor_free_warp(predictor, expected):
    path = Path(__file__).resolve().parents[1] / 'experiments/uavflow_predictor_idm/train.py'
    tree = ast.parse(path.read_text())
    expressions = [node.value for node in ast.walk(tree)
                   if isinstance(node, ast.FormattedValue)
                   and 'raw_model.predictor' in ast.unparse(node.value)
                   and 'variable_language_tokens' in ast.unparse(node.value)]
    assert len(expressions) == 1
    expression = ast.Expression(body=expressions[0])
    result = eval(compile(expression, str(path), 'eval'),
                  {'raw_model': SimpleNamespace(predictor=predictor)})
    assert result is expected
