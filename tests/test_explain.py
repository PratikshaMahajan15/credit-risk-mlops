from src.llm_explain import template_explanation, validate


REASONS = [{"label": "Debt-to-income ratio"}, {"label": "Length of credit history"}]


def test_template_mentions_reasons_and_disclaimer():
    text = template_explanation(612, "approve", REASONS)
    assert "612" in text and "debt-to-income ratio" in text and "does not change" in text


def test_validator_rejects_unsafe_or_made_up_text():
    ok, _ = validate("Your score is 612 because of the debt-to-income ratio.", 612, REASONS)
    assert ok
    assert not validate("This is a guarantee you will be fine.", 612, REASONS)[0]
    assert not validate("Your score is 700 due to debt-to-income ratio.", 612, REASONS)[0]
    assert not validate("Your score is 612 because of your age.", 612, REASONS)[0]
    assert not validate("Your score is 612.", 612, REASONS)[0]        # mentions no factor
