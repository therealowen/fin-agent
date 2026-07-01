"""
Agent 核心循环（ReAct 式）。

一次 agent.run(question) 内部会循环：
  1) 把对话历史 + 工具 schema 发给 LLM
  2) LLM 要么给最终答案（结束），要么要求调用某个工具
  3) 若要调工具：本地执行 -> 结果作为 tool 消息追加回历史 -> 回到 1)
  4) max_steps 上限，防止无限循环（面试常问：怎么保证 loop 会终止？）

供应商（DeepSeek / OpenAI）由 config.py 决定，本文件不关心用的是哪家。
"""
from __future__ import annotations
import json
import time

from openai import OpenAI

from config import get_provider
from tools import TOOL_SCHEMAS, run_tool

SYSTEM_PROMPT = (
    "你是一个金融数据助手。你可以调用工具获取行情、历史价格并做计算。"
    "遵循原则：需要数据时先调用工具，不要凭空编造数字；"
    "多步任务按顺序调用工具（如先取历史价格再算收益率）；"
    "拿到足够信息后用简洁中文给出最终答案，并说明关键数字的来源。"
)


class FinanceAgent:
    def __init__(self, max_steps: int = 6, verbose: bool = True):
        self.provider = get_provider()
        self.client = OpenAI(
            api_key=self.provider.api_key,
            base_url=self.provider.base_url,  # DeepSeek 走这里；OpenAI 时为 None=官方地址
        )
        self.model = self.provider.model
        self.max_steps = max_steps
        self.verbose = verbose
        if self.verbose:
            print(f"[provider] {self.provider.name} · model={self.model}")

    def _chat(self, messages: list[dict], retries: int = 2):
        """带简单重试的 LLM 调用（外部依赖要容错）。"""
        for attempt in range(retries + 1):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=0,
                )
            except Exception as e:
                if attempt == retries:
                    raise
                if self.verbose:
                    print(f"  [warn] LLM 调用失败，重试 {attempt + 1}/{retries}: {e}")
                time.sleep(1.5 * (attempt + 1))

    def run(self, question: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        for step in range(1, self.max_steps + 1):
            resp = self._chat(messages)
            msg = resp.choices[0].message

            # 情况 A：LLM 没要求调工具 -> 这是最终答案，循环结束
            if not msg.tool_calls:
                if self.verbose:
                    print(f"  [step {step}] 最终答案")
                return msg.content or ""

            # 情况 B：LLM 要求调用一个或多个工具
            messages.append(msg.model_dump())  # 先把 assistant 的 tool_calls 记进历史
            for call in msg.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments or "{}")
                if self.verbose:
                    print(f"  [step {step}] 调用工具 {name}({args})")
                result = run_tool(name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                })
            # 带着工具结果回到循环顶部，让 LLM 决定下一步

        # 到达步数上限仍未收敛 -> 明确降级，而不是硬编一个答案
        return "（已达到最大步数上限，未能完成推理。请缩小问题范围或稍后重试。）"


if __name__ == "__main__":
    agent = FinanceAgent()
    # q = "AAPL 最近 30 个交易日的收益率和年化波动率是多少？"
    # q = "AAPL 和 MSFT 最近 30 个交易日的表现对比,相关性高吗?"
    q = "AAPL 最近 30 天收益率是多少?再单独告诉我 MSFT 现在的收盘价。"
    print("Q:", q)
    print("A:", agent.run(q))
