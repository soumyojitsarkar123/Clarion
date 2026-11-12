from types import SimpleNamespace

import pytest

from services.token_budget_service import TokenBudgetService, TokenBudgetExceeded


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False, truncation=False):
        return text.split()


class SpecialTokenTokenizer(FakeTokenizer):
    def encode(self, text, add_special_tokens=False, truncation=False):
        tokens = super().encode(text, add_special_tokens=False, truncation=truncation)
        return ["<s>", *tokens, "</s>"] if add_special_tokens else tokens


def test_exact_count_and_budget_boundary():
    service = TokenBudgetService(10, 3, safety_tokens=2, tokenizer=FakeTokenizer())
    assert service.count("one two three four five").value == 5
    assert service.count("one two").exact is True
    assert service.plan("one two three four five").fits is True
    assert service.plan("one two three four six seven").fits is False
    with pytest.raises(TokenBudgetExceeded):
        service.require_fits("one two three four six seven")


def test_model_added_special_tokens_are_included_in_budget():
    service = TokenBudgetService(7, 2, safety_tokens=1, tokenizer=SpecialTokenTokenizer())
    assert service.count("one two").value == 4
    assert service.plan("one two").fits is True
    assert service.plan("one two three").fits is False


def test_missing_tokenizer_uses_explicit_utf8_estimate():
    service = TokenBudgetService(100, 10, safety_tokens=5)
    count = service.count("café")
    assert count.value == len("café".encode("utf-8"))
    assert count.exact is False
    assert count.method == "utf8_byte_upper_bound_estimate"


def test_ranked_packer_deduplicates_and_reports_whole_item_omissions():
    service = TokenBudgetService(40, 10, safety_tokens=5, tokenizer=FakeTokenizer())
    items = [
        SimpleNamespace(chunk_id="a", section_title="intro", content="alpha beta"),
        SimpleNamespace(chunk_id="b", section_title=None, content=" alpha   beta "),
        SimpleNamespace(chunk_id="c", section_title=None, content="gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega"),
    ]
    packed = service.pack_ranked_evidence("Query fixed text. Context:\n{context}\nAnswer:", items)
    assert packed.included_ids == ("a",)
    assert packed.omitted[0] == {"id": "b", "reason": "duplicate_content"}
    assert packed.omitted[1] == {"id": "c", "reason": "context_budget"}
    assert "alpha beta" in packed.context
    assert "gamma" not in packed.context
    assert packed.prompt_plan.fits


def test_invalid_budget_configuration_is_rejected():
    with pytest.raises(ValueError):
        TokenBudgetService(20, 15, safety_tokens=5)


def test_llm_rejects_over_budget_before_provider_call(monkeypatch):
    from services.knowledge_map_service import LLMInterface

    llm = LLMInterface()
    llm.token_budget = TokenBudgetService(20, 10, safety_tokens=5)
    monkeypatch.setattr(llm, "_get_client", lambda: pytest.fail("provider must not be called"))
    response = llm._call_api("x" * 10)
    assert "exceeded configured token budget" in response
    assert llm.last_token_metrics["request_rejected"] is True
    assert llm.last_token_metrics["preflight_exact"] is False
