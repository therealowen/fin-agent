
from __future__ import annotations
import io
import contextlib

from agent import FinanceAgent

# 每条：问题 + 答案里期望出现的关键词（宽松检查）
CASES = [
    {"q": "AAPL 现在的收盘价是多少？", "expect_any": ["收盘", "价", "$", "美元"]},
    {"q": "MSFT 最近 30 个交易日收益率是多少？", "expect_any": ["收益", "%"]},
    {"q": "帮我比较 AAPL 和 MSFT 最近一个月的表现", "expect_any": ["AAPL", "MSFT"]},
]


def run_eval():
    agent = FinanceAgent(verbose=False)
    passed = 0
    for i, case in enumerate(CASES, 1):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ans = agent.run(case["q"])
        ok = any(kw.lower() in ans.lower() for kw in case["expect_any"])
        passed += ok
        print(f"[{i}] {'PASS' if ok else 'FAIL'}  Q: {case['q']}")
        print(f"      A: {ans[:120]}{'...' if len(ans) > 120 else ''}")
    print(f"\n通过率: {passed}/{len(CASES)} = {passed / len(CASES):.0%}")


if __name__ == "__main__":
    run_eval()
