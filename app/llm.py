"""LLM 客户端。同样做成接口，测试注入 FakeLLM。"""

from __future__ import annotations

from typing import Protocol, Sequence


class LLM(Protocol):
    def chat(self, prompt: str) -> str:
        ...


class FakeLLM:
    """测试替身：记录被问过的 prompt，按剧本返回答案。"""

    def __init__(self, replies: Sequence[str] | None = None) -> None:
        self.prompts: list[str] = []
        self.replies = list(replies or ["（测试模型的回答）"])

    def chat(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if len(self.replies) > 1:
            return self.replies.pop(0)
        return self.replies[0]


class OpenAIChat:
    """OpenAI / DeepSeek 等 OpenAI 兼容接口。"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
