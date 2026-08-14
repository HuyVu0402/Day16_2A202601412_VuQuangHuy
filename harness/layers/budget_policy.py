"""Budget control layer.

This layer only controls tool spending. It nudges the model to produce a FINAL
when the useful tool budget is spent, and blocks extra tool calls once the
remaining budget is reserved for submit.

It does not fix wrong citations or hallucinated claims. Those belong to
CitationChecker and Critic.
"""

from __future__ import annotations

from arena.model import FINALIZE_SENTINEL, is_degraded
from arena.tools import ToolResult

from harness.middleware import Middleware

DEFAULT_RESERVE = 1

NUDGE = (
    "Ngan sach cong cu da het. Hay tra loi ngay bang FINAL tu bang chung dang co, "
    "khong goi them cong cu nao nua. "
    f"{FINALIZE_SENTINEL}"
)


class BudgetPolicy(Middleware):
    """Force the model to finish when the tool budget reaches the reserve."""

    name = "budget_policy"

    def __init__(self, reserve: int = DEFAULT_RESERVE) -> None:
        self.reserve = max(0, int(reserve))

    def _spent(self, ctx) -> bool:
        limit = ctx.max_tool_calls
        if is_degraded(ctx.observed_text):
            return False
        return limit is not None and ctx.tools.calls >= limit - self.reserve

    def before_model(self, ctx, messages):
        if not self._spent(ctx):
            return messages
        return messages + [{"role": "user", "content": NUDGE}]

    def wrap_tool_call(self, ctx, call, name, args):
        if not self._spent(ctx):
            return call(name, args)
        return ToolResult(
            ok=False,
            content="",
            error="tool budget reserved for submit",
        )
