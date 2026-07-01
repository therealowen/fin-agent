"""
Provider 配置层：一个开关切换 DeepSeek / OpenAI。

原理：DeepSeek 的 API 是 "OpenAI-compatible" 的——请求/响应格式与 OpenAI 一致，
只是 base_url 和模型名不同。所以我们用同一个 openai SDK，只切换 base_url + model + key。
这层抽象让"换模型供应商"变成改一个环境变量的事，是很实在的工程点。
"""
from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass
class ProviderConfig:
    name: str
    base_url: str | None      # None = 用 OpenAI 官方默认地址
    model: str
    api_key_env: str          # 从哪个环境变量读 key

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"未找到环境变量 {self.api_key_env}。"
                f"请先设置：export {self.api_key_env}=你的key"
            )
        return key


PROVIDERS = {
    "deepseek": ProviderConfig(
        name="deepseek",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
    ),
    "openai": ProviderConfig(
        name="openai",
        base_url=None,
        model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
    ),
}


def get_provider() -> ProviderConfig:
    """用环境变量 LLM_PROVIDER 选供应商，默认 deepseek。"""
    choice = os.environ.get("LLM_PROVIDER", "deepseek").lower()
    if choice not in PROVIDERS:
        raise ValueError(f"未知 provider：{choice}，可选 {list(PROVIDERS)}")
    return PROVIDERS[choice]
