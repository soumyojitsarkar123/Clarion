"""Conservative prompt accounting and ranked evidence packing.

When a model tokenizer is supplied, counts are exact for that tokenizer. Without
one, UTF-8 byte length is used as a conservative estimate (plus a configurable
template/safety reserve); it is explicitly marked estimated, never described as
provider token usage.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Optional
import re


@dataclass(frozen=True)
class TokenCount:
    value: int
    exact: bool
    method: str


@dataclass(frozen=True)
class BudgetPlan:
    context_limit: int
    input_tokens: int
    reserved_output_tokens: int
    safety_tokens: int
    remaining_tokens: int
    fits: bool
    exact: bool
    counting_method: str


@dataclass(frozen=True)
class PackedEvidence:
    context: str
    included_ids: tuple[str, ...]
    omitted: tuple[dict[str, str], ...]
    prompt_plan: BudgetPlan


class TokenBudgetExceeded(ValueError):
    """Raised when a request cannot fit without truncation."""


class ModelTokenBudgetService:
    def __init__(
        self,
        context_limit: int,
        reserved_output_tokens: int,
        safety_tokens: int = 256,
        tokenizer: Optional[Any] = None,
    ):
        if context_limit <= 0 or reserved_output_tokens < 0 or safety_tokens < 0:
            raise ValueError("Token budget values must be positive (reserves may be zero)")
        if reserved_output_tokens + safety_tokens >= context_limit:
            raise ValueError("Output and safety reserves must be smaller than context limit")
        self.context_limit = int(context_limit)
        self.reserved_output_tokens = int(reserved_output_tokens)
        self.safety_tokens = int(safety_tokens)
        self.tokenizer = tokenizer

    def count(self, text: str) -> TokenCount:
        text = text or ""
        if self.tokenizer is not None:
            try:
                # Include tokenizer-added BOS/EOS and other model framing tokens.
                encoded = self.tokenizer.encode(text, add_special_tokens=True, truncation=False)
                return TokenCount(len(encoded), True, "configured_model_tokenizer")
            except TypeError:
                encoded = self.tokenizer.encode(text)
                return TokenCount(len(encoded), True, "configured_model_tokenizer")
        # Byte fallback is intentionally conservative for byte-fallback tokenizers.
        return TokenCount(len(text.encode("utf-8")), False, "utf8_byte_upper_bound_estimate")

    def plan(self, prompt: str) -> BudgetPlan:
        count = self.count(prompt)
        remaining = self.context_limit - count.value - self.reserved_output_tokens - self.safety_tokens
        return BudgetPlan(
            context_limit=self.context_limit,
            input_tokens=count.value,
            reserved_output_tokens=self.reserved_output_tokens,
            safety_tokens=self.safety_tokens,
            remaining_tokens=remaining,
            fits=remaining >= 0,
            exact=count.exact,
            counting_method=count.method,
        )

    def require_fits(self, prompt: str) -> BudgetPlan:
        plan = self.plan(prompt)
        if not plan.fits:
            raise TokenBudgetExceeded(
                f"Prompt budget exceeded: input={plan.input_tokens}, output reserve="
                f"{plan.reserved_output_tokens}, safety={plan.safety_tokens}, "
                f"context={plan.context_limit} ({plan.counting_method})"
            )
        return plan

    def pack_ranked_evidence(
        self,
        fixed_prompt: str,
        evidence: Iterable[Any],
        *,
        id_field: str = "chunk_id",
    ) -> PackedEvidence:
        """Greedily keep ranked evidence that fits; never cut a source item."""
        included: list[str] = []
        included_ids: list[str] = []
        omitted: list[dict[str, str]] = []
        seen_content: set[str] = set()

        for rank, item in enumerate(evidence, 1):
            get = item.get if isinstance(item, dict) else lambda key, default=None: getattr(item, key, default)
            item_id = str(get(id_field, "") or f"rank-{rank}")
            content = str(get("content", "") or "")
            normalized = re.sub(r"\s+", " ", content).strip().casefold()
            if normalized and normalized in seen_content:
                omitted.append({"id": item_id, "reason": "duplicate_content"})
                continue
            header = f"[Evidence {rank}; id={item_id}]"
            section = str(get("section_title", "") or "").strip()
            if section:
                header += f" ({section})"
            candidate = "\n\n".join([*included, f"{header}\n{content}"])
            plan = self.plan(fixed_prompt.replace("{context}", candidate))
            if not plan.fits:
                omitted.append({"id": item_id, "reason": "context_budget"})
                continue
            included.append(f"{header}\n{content}")
            included_ids.append(item_id)
            if normalized:
                seen_content.add(normalized)

        context = "\n\n".join(included)
        final_plan = self.require_fits(fixed_prompt.replace("{context}", context))
        return PackedEvidence(context, tuple(included_ids), tuple(omitted), final_plan)


# Backwards-compatible public name; both names refer to the single service.
TokenBudgetService = ModelTokenBudgetService
