from src.rag.evaluate_gate import run_quality_gate


def test_evaluate_gate():
    assert run_quality_gate() is True
