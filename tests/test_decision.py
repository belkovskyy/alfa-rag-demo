from alfa_rag.decision import ALLOWED_STATUS, decision_policy_rules


def base_meta(**over):
    meta = {
        "intent": "faq",
        "top_score": 0.90,
        "overlap": 0.50,
        "n_base": 5,
        "n_keys": 3,
        "has_product_markers": True,
    }
    meta.update(over)
    return meta


def test_status_is_always_allowed():
    for meta in [base_meta(), base_meta(intent="personal"), base_meta(top_score=0.1), {}]:
        status, message = decision_policy_rules(meta)
        assert status in ALLOWED_STATUS
        assert isinstance(message, str) and message


def test_strong_retrieval_gives_ok():
    assert decision_policy_rules(base_meta())[0] == "ok"


def test_personal_intent_forces_clarify():
    assert decision_policy_rules(base_meta(intent="personal"))[0] == "need_clarify"


def test_weak_retrieval_gives_no_answer():
    assert decision_policy_rules(base_meta(top_score=0.10, n_base=1))[0] == "no_answer"


def test_retrieval_error_short_circuits():
    status, message = decision_policy_rules({"retrieval_error": "boom"})
    assert status == "no_answer"
    assert "boom" in message


def test_problem_without_product_is_not_ok():
    meta = base_meta(has_problem_markers=True, has_product_markers=False)
    assert decision_policy_rules(meta)[0] != "ok"
