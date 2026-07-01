"""
工具层：每个工具 = 一个普通 Python 函数 + 一份给 LLM 看的 JSON schema。
面试要点：工具是"能力边界"，schema 是"你如何把能力描述给模型"。
"""
from __future__ import annotations
import json
import statistics
from typing import Any

import yfinance as yf
import numpy as np


# ---------- 具体工具实现 ----------

def get_stock_price(ticker: str) -> dict[str, Any]:
    """取最近一个交易日的收盘价与基本信息。"""
    t = yf.Ticker(ticker)
    hist = t.history(period="5d")
    if hist.empty:
        return {"error": f"未找到 {ticker} 的行情数据"}
    last = hist.iloc[-1]
    return {
        "ticker": ticker.upper(),
        "date": str(hist.index[-1].date()),
        "close": round(float(last["Close"]), 2),
        "volume": int(last["Volume"]),
    }


def get_price_history(ticker: str, days: int = 30) -> dict[str, Any]:
    """取最近 N 个交易日的收盘价序列（给下游做计算用）。"""
    period = "1mo" if days <= 30 else "3mo" if days <= 90 else "1y"
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty:
        return {"error": f"未找到 {ticker} 的历史数据"}
    closes = [round(float(x), 2) for x in hist["Close"].tolist()][-days:]
    return {"ticker": ticker.upper(), "days": len(closes), "closes": closes}


def compute_return(closes: list[float]) -> dict[str, Any]:
    """给定收盘价序列，算区间收益率与年化波动率。"""
    if not closes or len(closes) < 2:
        return {"error": "收盘价序列过短，至少需要 2 个点"}
    total_return = (closes[-1] / closes[0] - 1) * 100
    daily_rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
    vol = statistics.pstdev(daily_rets) * (252 ** 0.5) * 100  # 年化波动率(%)
    return {
        "total_return_pct": round(total_return, 2),
        "annualized_vol_pct": round(vol, 2),
        "n_points": len(closes),
    }

def compare_stocks(ticker_a: str, ticker_b: str, days: int = 30) -> dict[str, Any]:
    """对比两只股票最近 N 个交易日的表现：各自收益率，以及两者日收益的相关系数。"""
    a = get_price_history(ticker_a, days)
    b = get_price_history(ticker_b, days)
    if "error" in a:
        return {"error": f"{ticker_a}: {a['error']}"}
    if "error" in b:
        return {"error": f"{ticker_b}: {b['error']}"}

    ca, cb = a["closes"], b["closes"]
    n = min(len(ca), len(cb))          # 两只股票交易日可能不完全对齐，取较短的
    if n < 2:
        return {"error": "重叠的交易日过少，无法计算相关性"}
    ca, cb = ca[-n:], cb[-n:]

    ret_a = np.diff(ca) / ca[:-1]      # 日收益率序列
    ret_b = np.diff(cb) / cb[:-1]
    corr = float(np.corrcoef(ret_a, ret_b)[0, 1])   # 皮尔逊相关系数

    return {
        "ticker_a": ticker_a.upper(),
        "ticker_b": ticker_b.upper(),
        "days": n,
        "return_a_pct": round((ca[-1] / ca[0] - 1) * 100, 2),
        "return_b_pct": round((cb[-1] / cb[0] - 1) * 100, 2),
        "daily_return_correlation": round(corr, 3),
    }

# ---------- 给 LLM 的 schema（function-calling 规范）----------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "获取某只股票最近一个交易日的收盘价、成交量。输入美股/常见 ticker，如 AAPL、MSFT。",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "股票代码，如 AAPL"}
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price_history",
            "description": "获取某只股票最近 N 个交易日的收盘价序列，用于后续计算收益率或波动率。",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "股票代码"},
                    "days": {"type": "integer", "description": "回溯的交易日数，默认 30"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_return",
            "description": "给定一个收盘价数组，计算区间总收益率(%)和年化波动率(%)。通常先用 get_price_history 拿到 closes 再调用它。",
            "parameters": {
                "type": "object",
                "properties": {
                    "closes": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "收盘价数组，按时间升序",
                    }
                },
                "required": ["closes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_stocks",
            "description": "对比两只股票最近 N 个交易日的表现：分别给出区间收益率，以及两者日收益率的相关系数（衡量走势同步程度，用于分散化分析）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_a": {"type": "string", "description": "第一只股票代码，如 AAPL"},
                    "ticker_b": {"type": "string", "description": "第二只股票代码，如 MSFT"},
                    "days": {"type": "integer", "description": "回溯的交易日数，默认 30"},
                },
                "required": ["ticker_a", "ticker_b"],
            },
        },
    },
]

# 名字 -> 函数 的分发表：agent 拿到 LLM 给的工具名后，用它找到真正要执行的函数
TOOL_REGISTRY = {
    "get_stock_price": get_stock_price,
    "get_price_history": get_price_history,
    "compute_return": compute_return,
    "compare_stocks": compare_stocks,
}


def run_tool(name: str, args: dict[str, Any]) -> str:
    """按名字执行工具，带最基础的错误兜底；返回 JSON 字符串喂回给 LLM。"""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return json.dumps({"error": f"未知工具 {name}"}, ensure_ascii=False)
    try:
        result = fn(**args)
    except Exception as e:  # 兜底：工具炸了不要让整个 loop 崩
        result = {"error": f"{name} 执行失败: {e}"}
    return json.dumps(result, ensure_ascii=False)
